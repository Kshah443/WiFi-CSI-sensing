/*
 * SPDX-FileCopyrightText: 2025-2026 Espressif Systems (Shanghai) CO LTD
 *
 * SPDX-License-Identifier: Apache-2.0
 */
/* Get Start Example

   This example code is in the Public Domain (or CC0 licensed, at your option.)

   Unless required by applicable law or agreed to in writing, this
   software is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
   CONDITIONS OF ANY KIND, either express or implied.
*/

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <inttypes.h>
#include <errno.h>
#include <fcntl.h>
#include <unistd.h>
#include "freertos/FreeRTOS.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"
#include "freertos/task.h"
#include "lwip/sockets.h"
#include "lwip/inet.h"

#include "nvs_flash.h"

#include "esp_mac.h"
#include "rom/ets_sys.h"
#include "esp_log.h"
#include "esp_wifi.h"
#include "esp_netif.h"
#include "esp_now.h"
#include "esp_csi_gain_ctrl.h"

#define CONFIG_LESS_INTERFERENCE_CHANNEL   11
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61 || (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0))
#define CONFIG_WIFI_BAND_MODE               WIFI_BAND_MODE_2G_ONLY
#define CONFIG_WIFI_2G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_5G_BANDWIDTHS           WIFI_BW_HT40
#define CONFIG_WIFI_2G_PROTOCOL             WIFI_PROTOCOL_11N
#define CONFIG_WIFI_5G_PROTOCOL             WIFI_PROTOCOL_11N
#else
#define CONFIG_WIFI_BANDWIDTH           WIFI_BW_HT40
#endif

#define CONFIG_ESP_NOW_PHYMODE           WIFI_PHY_MODE_HT40
#define CONFIG_ESP_NOW_RATE             WIFI_PHY_RATE_MCS0_LGI
#define CONFIG_FORCE_GAIN                   0

#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
#define CSI_FORCE_LLTF                      0
#endif

#if CONFIG_IDF_TARGET_ESP32S3 || CONFIG_IDF_TARGET_ESP32C3 || CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
#define CONFIG_GAIN_CONTROL                 1
#endif

#if ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(6, 0, 0)
#define ESP_IF_WIFI_STA ESP_MAC_WIFI_STA
#endif

static const uint8_t CONFIG_CSI_SEND_MAC[] = {0x1a, 0x00, 0x00, 0x00, 0x00, 0x00};
static const char *TAG = "csi_recv";

/* Fixed storage: no allocation, printing, or socket operations in the callback. */
#define CSI_MAX_BYTES 2048
#define CSI_QUEUE_DEPTH 32
#define UDP_HEADER_BYTES 16

typedef struct {
    wifi_csi_info_t info;
    uint32_t sequence;
    float compensate_gain;
    uint8_t agc_gain;
    int8_t fft_gain;
    int8_t data[CSI_MAX_BYTES];
} csi_packet_t;

_Static_assert(sizeof(CONFIG_CSI_UDP_AP_SSID) > 1 && sizeof(CONFIG_CSI_UDP_AP_SSID) <= 33,
               "SoftAP SSID must be 1 to 32 bytes");
_Static_assert(sizeof(CONFIG_CSI_UDP_AP_PASSWORD) >= 9 && sizeof(CONFIG_CSI_UDP_AP_PASSWORD) <= 64,
               "SoftAP password must be 8 to 63 bytes");

static QueueHandle_t s_csi_queue;
static SemaphoreHandle_t s_serial_lock;
static portMUX_TYPE s_stats_lock = portMUX_INITIALIZER_UNLOCKED;
static struct {
    uint32_t received, queued, sent, dropped;
} s_stats;
static int s_udp_socket = -1;
static struct sockaddr_in s_destination;

static void put_u16(uint8_t *out, uint16_t value)
{
    out[0] = value >> 8;
    out[1] = value;
}

static void put_u32(uint8_t *out, uint32_t value)
{
    out[0] = value >> 24;
    out[1] = value >> 16;
    out[2] = value >> 8;
    out[3] = value;
}

/* Network byte order, explicit offsets: never transmit a padded C struct. */
static void csi_udp_send(const wifi_csi_info_t *info, uint32_t sequence, float gain)
{
    static uint8_t datagram[UDP_HEADER_BYTES + CSI_MAX_BYTES * 2];
    memcpy(datagram, "CSI1", 4);
    put_u32(datagram + 4, sequence);
    datagram[8] = (uint8_t)info->rx_ctrl.rssi;
    datagram[9] = info->first_word_invalid;
    datagram[10] = info->rx_ctrl.channel;
    datagram[11] = 0;
    put_u16(datagram + 12, info->len);
    uint16_t count = 0;
#if (CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61) && CSI_FORCE_LLTF
    for (int i = 0; i < (info->len - 2); i += 2) {
        int16_t csi = ((int16_t)(((((uint16_t)info->buf[i + 1]) << 8) | info->buf[i]) << 4) >> 4);
        put_u16(datagram + UDP_HEADER_BYTES + 2 * count++, (uint16_t)(int16_t)(gain * csi));
    }
#else
    for (int i = 0; i < info->len; ++i) {
        put_u16(datagram + UDP_HEADER_BYTES + 2 * count++, (uint16_t)(int16_t)(gain * info->buf[i]));
    }
#endif
    put_u16(datagram + 14, count);
    size_t size = UDP_HEADER_BYTES + 2 * count;
    int result = -1;
    if (s_udp_socket >= 0) {
        result = sendto(s_udp_socket, datagram, size, MSG_DONTWAIT,
                        (struct sockaddr *)&s_destination, sizeof(s_destination));
    }
    portENTER_CRITICAL(&s_stats_lock);
    if (result == (int)size) {
        ++s_stats.sent;
    } else {
        ++s_stats.dropped;
    }
    portEXIT_CRITICAL(&s_stats_lock);
}

static void wifi_init()
{
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_netif_create_default_wifi_ap() ? ESP_OK : ESP_FAIL);
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_APSTA));
    ESP_ERROR_CHECK(esp_wifi_set_storage(WIFI_STORAGE_RAM));
    wifi_config_t ap_config = {
        .ap = {
            .ssid = CONFIG_CSI_UDP_AP_SSID,
            .password = CONFIG_CSI_UDP_AP_PASSWORD,
            .channel = CONFIG_LESS_INTERFERENCE_CHANNEL,
            .max_connection = 1,
            .authmode = WIFI_AUTH_WPA2_PSK,
        },
    };
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_AP, &ap_config));

#if CONFIG_IDF_TARGET_ESP32C5
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
        .ghz_5g = CONFIG_WIFI_5G_PROTOCOL
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
        .ghz_5g = CONFIG_WIFI_5G_BANDWIDTHS
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
    ESP_ERROR_CHECK(esp_wifi_set_protocols(WIFI_IF_AP, &protocols));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(WIFI_IF_AP, &bandwidth));
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    ESP_ERROR_CHECK(esp_wifi_start());
    esp_wifi_set_band_mode(CONFIG_WIFI_BAND_MODE);
    wifi_protocols_t protocols = {
        .ghz_2g = CONFIG_WIFI_2G_PROTOCOL,
    };
    ESP_ERROR_CHECK(esp_wifi_set_protocols(ESP_IF_WIFI_STA, &protocols));
    wifi_bandwidths_t bandwidth = {
        .ghz_2g = CONFIG_WIFI_2G_BANDWIDTHS,
    };
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(ESP_IF_WIFI_STA, &bandwidth));
    ESP_ERROR_CHECK(esp_wifi_set_protocols(WIFI_IF_AP, &protocols));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidths(WIFI_IF_AP, &bandwidth));
#else
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(ESP_IF_WIFI_STA, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_set_bandwidth(WIFI_IF_AP, CONFIG_WIFI_BANDWIDTH));
    ESP_ERROR_CHECK(esp_wifi_start());
#endif

    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
#if CONFIG_IDF_TARGET_ESP32C5
    if ((CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20)
            || (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_5G_ONLY && CONFIG_WIFI_5G_BANDWIDTHS == WIFI_BW_HT20)) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#elif (CONFIG_IDF_TARGET_ESP32C6 && ESP_IDF_VERSION >= ESP_IDF_VERSION_VAL(5, 4, 0)) || CONFIG_IDF_TARGET_ESP32C61
    if (CONFIG_WIFI_BAND_MODE == WIFI_BAND_MODE_2G_ONLY && CONFIG_WIFI_2G_BANDWIDTHS == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#else
    if (CONFIG_WIFI_BANDWIDTH == WIFI_BW_HT20) {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_NONE));
    } else {
        ESP_ERROR_CHECK(esp_wifi_set_channel(CONFIG_LESS_INTERFERENCE_CHANNEL, WIFI_SECOND_CHAN_BELOW));
    }
#endif

    ESP_ERROR_CHECK(esp_wifi_set_mac(WIFI_IF_STA, CONFIG_CSI_SEND_MAC));
}

static void wifi_esp_now_init(esp_now_peer_info_t peer)
{
    ESP_ERROR_CHECK(esp_now_init());
    ESP_ERROR_CHECK(esp_now_set_pmk((uint8_t *)"pmk1234567890123"));
    esp_now_rate_config_t rate_config = {
        .phymode = CONFIG_ESP_NOW_PHYMODE,
        .rate = CONFIG_ESP_NOW_RATE,//  WIFI_PHY_RATE_MCS0_LGI,
        .ersu = false,
        .dcm = false
    };
    ESP_ERROR_CHECK(esp_now_add_peer(&peer));
    ESP_ERROR_CHECK(esp_now_set_peer_rate_config(peer.peer_addr, &rate_config));

}

static void wifi_csi_rx_cb(void *ctx, wifi_csi_info_t *info)
{
    if (!info || !info->buf || memcmp(info->mac, CONFIG_CSI_SEND_MAC, 6)) {
        return;
    }
    portENTER_CRITICAL(&s_stats_lock);
    ++s_stats.received;
    portEXIT_CRITICAL(&s_stats_lock);

    if (!info->payload || info->payload_len < 19 || !info->len || info->len > CSI_MAX_BYTES) {
        portENTER_CRITICAL(&s_stats_lock);
        ++s_stats.dropped;
        portEXIT_CRITICAL(&s_stats_lock);
        return;
    }
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static unsigned s_gain_count;
    float compensate_gain = 1.0f;
    static uint8_t agc_gain = 0;
    static int8_t fft_gain = 0;
#if CONFIG_GAIN_CONTROL
    static uint8_t agc_gain_baseline = 0;
    static int8_t fft_gain_baseline = 0;
    esp_csi_gain_ctrl_get_rx_gain(rx_ctrl, &agc_gain, &fft_gain);
    if (s_gain_count < 100) {
        esp_csi_gain_ctrl_record_rx_gain(agc_gain, fft_gain);
    } else if (s_gain_count == 100) {
        esp_csi_gain_ctrl_get_rx_gain_baseline(&agc_gain_baseline, &fft_gain_baseline);
#if CONFIG_FORCE_GAIN
        esp_csi_gain_ctrl_set_rx_force_gain(agc_gain_baseline, fft_gain_baseline);
        ESP_LOGD(TAG, "fft_force %d, agc_force %d", fft_gain_baseline, agc_gain_baseline);
#endif
    }
    esp_csi_gain_ctrl_get_gain_compensation(&compensate_gain, agc_gain, fft_gain);
#endif

    ++s_gain_count;
    /* Wi-Fi invokes this callback serially. The queue copies the entire item. */
    static csi_packet_t packet;
    packet.info = *info;
    packet.info.buf = NULL;
    packet.info.payload = NULL;
    packet.info.hdr = NULL;
    memcpy(&packet.sequence, info->payload + 15, sizeof(packet.sequence));
    packet.compensate_gain = compensate_gain;
    packet.agc_gain = agc_gain;
    packet.fft_gain = fft_gain;
    memcpy(packet.data, info->buf, info->len);
    BaseType_t queued = xQueueSend(s_csi_queue, &packet, 0);
    portENTER_CRITICAL(&s_stats_lock);
    if (queued == pdTRUE) {
        ++s_stats.queued;
    } else {
        ++s_stats.dropped;
    }
    portEXIT_CRITICAL(&s_stats_lock);
}

static void csi_process_packet(csi_packet_t *packet)
{
    wifi_csi_info_t *info = &packet->info;
    info->buf = packet->data;
    const wifi_pkt_rx_ctrl_t *rx_ctrl = &info->rx_ctrl;
    static int s_count = 0;
    float compensate_gain = packet->compensate_gain;
    uint8_t agc_gain = packet->agc_gain;
    int8_t fft_gain = packet->fft_gain;
#if CONFIG_GAIN_CONTROL
    ESP_LOGI(TAG, "compensate_gain %f, agc_gain %d, fft_gain %d", compensate_gain, agc_gain, fft_gain);
#endif
    uint32_t rx_id = packet->sequence;
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C6 || CONFIG_IDF_TARGET_ESP32C61
    if (!s_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,seq,mac,rssi,rate,noise_floor,fft_gain,agc_gain,channel,local_timestamp,sig_len,rx_format,len,first_word,data\n");
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate,
               rx_ctrl->noise_floor, fft_gain, agc_gain,  rx_ctrl->channel,
               rx_ctrl->timestamp, rx_ctrl->sig_len, rx_ctrl->cur_bb_format);
#else
    if (!s_count) {
        ESP_LOGI(TAG, "================ CSI RECV ================");
        ets_printf("type,id,mac,rssi,rate,sig_mode,mcs,bandwidth,smoothing,not_sounding,aggregation,stbc,fec_coding,sgi,noise_floor,ampdu_cnt,channel,secondary_channel,local_timestamp,ant,sig_len,rx_format,len,first_word,data\n");
    }

    ets_printf("CSI_DATA,%d," MACSTR ",%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d",
               rx_id, MAC2STR(info->mac), rx_ctrl->rssi, rx_ctrl->rate, rx_ctrl->sig_mode,
               rx_ctrl->mcs, rx_ctrl->cwb, rx_ctrl->smoothing, rx_ctrl->not_sounding,
               rx_ctrl->aggregation, rx_ctrl->stbc, rx_ctrl->fec_coding, rx_ctrl->sgi,
               rx_ctrl->noise_floor, rx_ctrl->ampdu_cnt, rx_ctrl->channel, rx_ctrl->secondary_channel,
               rx_ctrl->timestamp, rx_ctrl->ant, rx_ctrl->sig_len, rx_ctrl->sig_mode);

#endif
#if (CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61) && CSI_FORCE_LLTF
    int16_t csi = ((int16_t)(((((uint16_t)info->buf[1]) << 8) | info->buf[0]) << 4) >> 4);
    ets_printf(",%d,%d,\"[%d", (info->len - 2) / 2, info->first_word_invalid, (int16_t)(compensate_gain * csi));
    for (int i = 2; i < (info->len - 2); i += 2) {
        csi = ((int16_t)(((((uint16_t)info->buf[i + 1]) << 8) | info->buf[i]) << 4) >> 4);
        ets_printf(",%d", (int16_t)(compensate_gain * csi));
    }
#else
    ets_printf(",%d,%d,\"[%d", info->len, info->first_word_invalid, (int16_t)(compensate_gain * info->buf[0]));
    for (int i = 1; i < info->len; i++) {
        ets_printf(",%d", (int16_t)(compensate_gain * info->buf[i]));
    }
#endif
    ets_printf("]\"\n");
    s_count++;
    csi_udp_send(info, rx_id, compensate_gain);
}

static void csi_worker(void *arg)
{
    static csi_packet_t packet;
    for (;;) {
        if (s_udp_socket < 0) {
            s_udp_socket = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
            if (s_udp_socket >= 0 && fcntl(s_udp_socket, F_SETFL, O_NONBLOCK) < 0) {
                close(s_udp_socket);
                s_udp_socket = -1;
            }
        }
        if (xQueueReceive(s_csi_queue, &packet, pdMS_TO_TICKS(1000)) == pdTRUE) {
            xSemaphoreTake(s_serial_lock, portMAX_DELAY);
            csi_process_packet(&packet);
            xSemaphoreGive(s_serial_lock);
        }
    }
}

static void csi_stats_task(void *arg)
{
    for (;;) {
        vTaskDelay(pdMS_TO_TICKS(5000));
        uint32_t received, queued, sent, dropped;
        portENTER_CRITICAL(&s_stats_lock);
        received = s_stats.received;
        queued = s_stats.queued;
        sent = s_stats.sent;
        dropped = s_stats.dropped;
        portEXIT_CRITICAL(&s_stats_lock);
        xSemaphoreTake(s_serial_lock, portMAX_DELAY);
        ESP_LOGI(TAG, "CSI_STATS received=%" PRIu32 " queued=%" PRIu32
                 " udp_sent=%" PRIu32 " dropped=%" PRIu32,
                 received, queued, sent, dropped);
        xSemaphoreGive(s_serial_lock);
    }
}

static void csi_forwarding_init(void)
{
    s_destination.sin_family = AF_INET;
    s_destination.sin_port = htons(CONFIG_CSI_UDP_DEST_PORT);
    ESP_ERROR_CHECK(inet_pton(AF_INET, CONFIG_CSI_UDP_DEST_IP,
                             &s_destination.sin_addr) == 1 ? ESP_OK : ESP_ERR_INVALID_ARG);
    s_serial_lock = xSemaphoreCreateMutex();
    ESP_ERROR_CHECK(s_serial_lock ? ESP_OK : ESP_ERR_NO_MEM);
    s_csi_queue = xQueueCreate(CSI_QUEUE_DEPTH, sizeof(csi_packet_t));
    ESP_ERROR_CHECK(s_csi_queue ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_ERROR_CHECK(xTaskCreate(csi_worker, "csi_forward", 6144, NULL, 4, NULL) == pdPASS
                    ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_ERROR_CHECK(xTaskCreate(csi_stats_task, "csi_stats", 3072, NULL, 2, NULL) == pdPASS
                    ? ESP_OK : ESP_ERR_NO_MEM);
    ESP_LOGI(TAG, "UDP destination %s:%d; connect laptop to AP %s (channel %d)",
             CONFIG_CSI_UDP_DEST_IP, CONFIG_CSI_UDP_DEST_PORT,
             CONFIG_CSI_UDP_AP_SSID, CONFIG_LESS_INTERFERENCE_CHANNEL);
}

static void wifi_csi_init()
{
    ESP_ERROR_CHECK(esp_wifi_set_promiscuous(true));

    /**< default config */
#if CONFIG_IDF_TARGET_ESP32C5 || CONFIG_IDF_TARGET_ESP32C61
    wifi_csi_config_t csi_config = {
        .enable                   = true,
        .acquire_csi_legacy       = false,
        .acquire_csi_force_lltf   = CSI_FORCE_LLTF,
        .acquire_csi_ht20         = true,
        .acquire_csi_ht40         = true,
        .acquire_csi_vht          = false,
        .acquire_csi_su           = false,
        .acquire_csi_mu           = false,
        .acquire_csi_dcm          = false,
        .acquire_csi_beamformed   = false,
        .acquire_csi_he_stbc_mode = 2,
        .val_scale_cfg            = 0,
        .dump_ack_en              = false,
        .reserved                 = false
    };
#elif CONFIG_IDF_TARGET_ESP32C6
    wifi_csi_config_t csi_config = {
        .enable                 = true,
        .acquire_csi_legacy     = false,
        .acquire_csi_ht20       = true,
        .acquire_csi_ht40       = true,
        .acquire_csi_su         = true,
        .acquire_csi_mu         = true,
        .acquire_csi_dcm        = true,
        .acquire_csi_beamformed = true,
        .acquire_csi_he_stbc    = 2,
        .val_scale_cfg          = false,
        .dump_ack_en            = false,
        .reserved               = false
    };
#else
    wifi_csi_config_t csi_config = {
        .lltf_en           = true,
        .htltf_en          = true,
        .stbc_htltf2_en    = true,
        .ltf_merge_en      = true,
        .channel_filter_en = true,
        .manu_scale        = false,
        .shift             = false,
    };
#endif
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_rx_cb, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));
}

void app_main()
{
    /**
     * @brief Initialize NVS
     */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /**
     * @brief Initialize Wi-Fi
     */
    wifi_init();

    /**
     * @brief Initialize ESP-NOW
     *        ESP-NOW protocol see: https://docs.espressif.com/projects/esp-idf/en/latest/esp32/api-reference/network/esp_now.html
     */

    esp_now_peer_info_t peer = {
        .channel   = CONFIG_LESS_INTERFERENCE_CHANNEL,
        .ifidx     = WIFI_IF_STA,
        .encrypt   = false,
        .peer_addr = {0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
    };

    wifi_esp_now_init(peer);

    csi_forwarding_init();
    wifi_csi_init();
}
