package com.example.yolopose.network

data class Esp32ProvisioningConfig(
    val autoProvisionEnabled: Boolean,
    val deviceNamePrefix: String,
    val hotspotSsid: String,
    val hotspotPassword: String,
    val lastKnownHost: String?
)

sealed class Esp32ProvisioningState {
    data object Scanning : Esp32ProvisioningState()
    data class DeviceFound(val deviceName: String) : Esp32ProvisioningState()
    data class DeviceConnected(val deviceName: String) : Esp32ProvisioningState()
    data object DiscoveringServices : Esp32ProvisioningState()
    data object ServicesReady : Esp32ProvisioningState()
    data class SendingHotspotConfig(val ssid: String) : Esp32ProvisioningState()
    data class WaitingForDeviceJoin(val ssid: String) : Esp32ProvisioningState()
    data class Completed(val esp32Host: String) : Esp32ProvisioningState()
    data class Failed(val message: String) : Esp32ProvisioningState()
}
