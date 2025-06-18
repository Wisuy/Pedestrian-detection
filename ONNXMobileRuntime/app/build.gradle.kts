plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.example.onnxmobileruntime"
    compileSdk = 34 // Or your desired compile SDK version

    defaultConfig {
        applicationId = "com.example.onnxmobileruntime"
        minSdk = 24 // Minimum SDK supported by ONNX Runtime Android
        targetSdk = 34 // Or your desired target SDK version
        versionCode = 1
        versionName = "1.0"

        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_1_8
        targetCompatibility = JavaVersion.VERSION_1_8
    }
    kotlinOptions {
        jvmTarget = "1.8"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // Core Android KTX
    implementation(libs.androidx.core.ktx.v1131)
    implementation(libs.androidx.appcompat)
    implementation(libs.material)
    implementation(libs.androidx.constraintlayout)

    // CameraX - Using the stable 1.2.3 version as previously recommended
    implementation(libs.androidx.camera.core)
    implementation(libs.androidx.camera.camera2)
    implementation(libs.androidx.camera.lifecycle)
    implementation(libs.androidx.camera.view)
    implementation(libs.androidx.camera.extensions)

    // ONNX Runtime Android
    implementation(libs.onnxruntime.android)

    // Lifecycle KTX for LiveData and ViewModel
    implementation(libs.androidx.lifecycle.runtime.ktx.v281)
    implementation(libs.androidx.lifecycle.viewmodel.ktx)

    // For testing
    testImplementation(libs.junit)
    androidTestImplementation(libs.androidx.junit.v115)
    androidTestImplementation(libs.androidx.espresso.core.v351)
}