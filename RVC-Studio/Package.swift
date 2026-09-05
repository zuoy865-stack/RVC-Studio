// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "RVCStudio",
    platforms: [.macOS(.v15)],
    products: [
        .executable(name: "RVCStudio", targets: ["RVCStudio"])
    ],
    targets: [
        .executableTarget(
            name: "RVCStudio",
            path: "Sources/RVCStudio",
            swiftSettings: [
                .swiftLanguageMode(.v5)
            ],
            linkerSettings: [
                .linkedFramework("AppKit"),
                .linkedFramework("AVFoundation")
            ]
        )
    ]
)
