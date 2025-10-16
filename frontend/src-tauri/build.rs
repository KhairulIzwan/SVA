fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Generate gRPC client code
    tonic_build::configure()
        .build_server(false)  // We only need client code
        .compile(&["../../proto/sva.proto"], &["../../proto"])?;
    
    // Standard Tauri build
    tauri_build::build();
    Ok(())
}
