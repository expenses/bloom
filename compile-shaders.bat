:: Keep this target up-to-date with the rust-gpu one.
:: rustup +nightly-2021-09-29-x86_64-pc-windows-msvc component add rust-src rustc-dev llvm-tools-preview
cargo +nightly-2021-09-29-x86_64-pc-windows-msvc run -p compile-shaders --release
