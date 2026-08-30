package com.example.yolopose.network

import android.content.Context
import android.security.keystore.KeyGenParameterSpec
import android.security.keystore.KeyProperties
import android.util.Base64
import java.nio.charset.StandardCharsets
import java.security.KeyStore
import javax.crypto.Cipher
import javax.crypto.KeyGenerator
import javax.crypto.SecretKey
import javax.crypto.spec.GCMParameterSpec

class Esp32ProvisioningStore(context: Context) {
    private val prefs = context.applicationContext.getSharedPreferences("esp32_provisioning_secure", Context.MODE_PRIVATE)

    fun load(): Esp32ProvisioningConfig {
        return Esp32ProvisioningConfig(
            autoProvisionEnabled = prefs.getBoolean(KEY_AUTO_ENABLED, true),
            deviceNamePrefix = prefs.getString(KEY_DEVICE_PREFIX, DEFAULT_DEVICE_PREFIX).orEmpty(),
            hotspotSsid = prefs.getString(KEY_HOTSPOT_SSID, DEFAULT_HOTSPOT_SSID).orEmpty(),
            hotspotPassword = decrypt(prefs.getString(KEY_HOTSPOT_PASSWORD, null)) ?: DEFAULT_HOTSPOT_PASSWORD,
            lastKnownHost = prefs.getString(KEY_LAST_HOST, DEFAULT_LAST_HOST)
        )
    }

    fun save(config: Esp32ProvisioningConfig) {
        prefs.edit()
            .putBoolean(KEY_AUTO_ENABLED, config.autoProvisionEnabled)
            .putString(KEY_DEVICE_PREFIX, config.deviceNamePrefix)
            .putString(KEY_HOTSPOT_SSID, config.hotspotSsid)
            .putString(KEY_HOTSPOT_PASSWORD, encrypt(config.hotspotPassword))
            .putString(KEY_LAST_HOST, config.lastKnownHost ?: DEFAULT_LAST_HOST)
            .apply()
    }

    fun updateLastKnownHost(host: String) {
        prefs.edit().putString(KEY_LAST_HOST, host).apply()
    }

    private fun encrypt(plainText: String): String {
        if (plainText.isBlank()) return ""
        val cipher = Cipher.getInstance(TRANSFORMATION)
        cipher.init(Cipher.ENCRYPT_MODE, getOrCreateSecretKey())
        val iv = cipher.iv
        val encrypted = cipher.doFinal(plainText.toByteArray(StandardCharsets.UTF_8))
        return Base64.encodeToString(iv + encrypted, Base64.NO_WRAP)
    }

    private fun decrypt(payload: String?): String? {
        if (payload.isNullOrBlank()) return null
        return runCatching {
            val allBytes = Base64.decode(payload, Base64.NO_WRAP)
            val iv = allBytes.copyOfRange(0, IV_LENGTH_BYTES)
            val cipherBytes = allBytes.copyOfRange(IV_LENGTH_BYTES, allBytes.size)
            val cipher = Cipher.getInstance(TRANSFORMATION)
            cipher.init(
                Cipher.DECRYPT_MODE,
                getOrCreateSecretKey(),
                GCMParameterSpec(GCM_TAG_LENGTH_BITS, iv)
            )
            String(cipher.doFinal(cipherBytes), StandardCharsets.UTF_8)
        }.getOrNull()
    }

    private fun getOrCreateSecretKey(): SecretKey {
        val keyStore = KeyStore.getInstance(ANDROID_KEY_STORE).apply { load(null) }
        val existingKey = keyStore.getKey(KEY_ALIAS, null) as? SecretKey
        if (existingKey != null) {
            return existingKey
        }
        val keyGenerator = KeyGenerator.getInstance(KeyProperties.KEY_ALGORITHM_AES, ANDROID_KEY_STORE)
        keyGenerator.init(
            KeyGenParameterSpec.Builder(
                KEY_ALIAS,
                KeyProperties.PURPOSE_ENCRYPT or KeyProperties.PURPOSE_DECRYPT
            )
                .setBlockModes(KeyProperties.BLOCK_MODE_GCM)
                .setEncryptionPaddings(KeyProperties.ENCRYPTION_PADDING_NONE)
                .setRandomizedEncryptionRequired(true)
                .build()
        )
        return keyGenerator.generateKey()
    }

    companion object {
        private const val KEY_AUTO_ENABLED = "auto_enabled"
        private const val KEY_DEVICE_PREFIX = "device_prefix"
        private const val KEY_HOTSPOT_SSID = "hotspot_ssid"
        private const val KEY_HOTSPOT_PASSWORD = "hotspot_password"
        private const val KEY_LAST_HOST = "last_host"

        private const val DEFAULT_DEVICE_PREFIX = "ESP32"
        private const val DEFAULT_HOTSPOT_SSID = ""
        private const val DEFAULT_HOTSPOT_PASSWORD = ""
        private const val DEFAULT_LAST_HOST = "192.168.4.1"

        private const val ANDROID_KEY_STORE = "AndroidKeyStore"
        private const val KEY_ALIAS = "esp32_provisioning_key"
        private const val TRANSFORMATION = "AES/GCM/NoPadding"
        private const val GCM_TAG_LENGTH_BITS = 128
        private const val IV_LENGTH_BYTES = 12
    }
}
