package com.example.yolopose.network

import android.content.Context
import android.net.ConnectivityManager
import android.net.Network
import android.net.NetworkCapabilities
import android.net.NetworkRequest
import android.net.wifi.WifiManager
import android.net.wifi.WifiNetworkSpecifier
import android.os.Build

class Esp32WifiConnector(
    context: Context,
    private val ssid: String,
    private val password: String?,
    private val onConnected: (Network) -> Unit,
    private val onUnavailable: (String) -> Unit
) {
    private val appContext = context.applicationContext
    private val connectivityManager =
        appContext.getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
    private val wifiManager =
        appContext.getSystemService(Context.WIFI_SERVICE) as WifiManager

    private var currentCallback: ConnectivityManager.NetworkCallback? = null
    var activeNetwork: Network? = null
        private set

    fun isAlreadyConnected(): Boolean {
        val ssidValue = wifiManager.connectionInfo?.ssid?.trim('"') ?: return false
        return ssidValue == ssid
    }

    fun supportsConcurrentDeviceLink(): Boolean {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            wifiManager.isStaConcurrencyForLocalOnlyConnectionsSupported
        } else {
            false
        }
    }

    fun connectIfNeeded() {
        if (isAlreadyConnected()) {
            connectivityManager.activeNetwork?.let {
                activeNetwork = it
                onConnected(it)
                return
            }
        }
        runCatching {
            requestNetwork()
        }.onFailure { error ->
            onUnavailable(error.message ?: error.javaClass.simpleName)
        }
    }

    fun release() {
        currentCallback?.let {
            runCatching { connectivityManager.unregisterNetworkCallback(it) }
        }
        currentCallback = null
        activeNetwork = null
    }

    private fun requestNetwork() {
        release()
        val builder = WifiNetworkSpecifier.Builder()
            .setSsid(ssid)
        if (!password.isNullOrBlank()) {
            builder.setWpa2Passphrase(password)
        }
        val specifier = builder.build()

        val request = NetworkRequest.Builder()
            .addTransportType(NetworkCapabilities.TRANSPORT_WIFI)
            .removeCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            .setNetworkSpecifier(specifier)
            .build()

        val callback = object : ConnectivityManager.NetworkCallback() {
            override fun onAvailable(network: Network) {
                activeNetwork = network
                onConnected(network)
            }

            override fun onUnavailable() {
                onUnavailable("系统未连接到 $ssid")
            }

            override fun onLost(network: Network) {
                if (activeNetwork == network) {
                    activeNetwork = null
                    onUnavailable("$ssid 已断开")
                }
            }
        }
        currentCallback = callback
        connectivityManager.requestNetwork(request, callback)
    }
}
