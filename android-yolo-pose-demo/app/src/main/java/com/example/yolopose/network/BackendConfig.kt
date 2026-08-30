package com.example.yolopose.network

import com.example.yolopose.BuildConfig

object BackendConfig {
    const val API_TIMEOUT_MS = 8_000L
    val baseUrl: String = BuildConfig.API_BASE_URL
    val demoAccount: String = BuildConfig.API_DEMO_ACCOUNT
    val demoPassword: String = BuildConfig.API_DEMO_PASSWORD
}
