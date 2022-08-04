use std::env;
use std::path::PathBuf;

fn main() {
    // account for cross-compilation [by examining environment variable]
    let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();

    // Set CXX environment variable to choose alternative C compiler.
    // Optimization level depends on whether or not --release is passed
    // or implied.
    let mut cc = cc::Build::new();
    cc.cpp(true);

    let c_src_dir = PathBuf::from("src");
    let files = vec![c_src_dir.join("pippenger.cpp")];
    let mut cc_def = None;

    match (cfg!(feature = "portable"), cfg!(feature = "force-adx")) {
        (true, false) => {
            println!("Compiling in portable mode without ISA extensions");
            cc_def = Some("__PASTA_PORTABLE__");
        }
        (false, true) => {
            if target_arch.eq("x86_64") {
                println!("Enabling ADX support via `force-adx` feature");
                cc_def = Some("__ADX__");
            } else {
                println!("`force-adx` is ignored for non-x86_64 targets");
            }
        }
        (false, false) => {
            #[cfg(target_arch = "x86_64")]
            if target_arch.eq("x86_64") && std::is_x86_feature_detected!("adx")
            {
                println!("Enabling ADX because it was detected on the host");
                cc_def = Some("__ADX__");
            }
        }
        (true, true) => panic!(
            "Cannot compile with both `portable` and `force-adx` features"
        ),
    }

    cc.flag_if_supported("-mno-avx") // avoid costly transitions
        .flag_if_supported("-fno-builtin")
        .flag_if_supported("-std=c++11")
        .flag_if_supported("-Wno-unused-command-line-argument");
    if !cfg!(debug_assertions) {
        cc.define("NDEBUG", None);
    }
    if let Some(def) = cc_def {
        cc.define(def, None);
    }
    if let Some(include) = env::var_os("DEP_SEMOLINA_C_INCLUDE") {
        cc.include(include);
    }
    if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
        cc.include(include);
    }
    cc.files(&files).compile("pasta_msm");

    if cfg!(target_os = "windows") && !cfg!(target_env = "msvc") {
        return;
    }
    // Detect if there is CUDA compiler and engage "cuda" feature accordingly
    let nvcc = match env::var("NVCC") {
        Ok(var) => which::which(var),
        Err(_) => which::which("nvcc"),
    };
    if nvcc.is_ok() {
        let mut nvcc = cc::Build::new();
        nvcc.cuda(true);
        nvcc.flag("-Xcompiler").flag("-Wno-unused-function");
        nvcc.flag("-arch=sm_70");
        nvcc.define("TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE", None);
        #[cfg(feature = "cuda-mobile")]
        nvcc.define("NTHREADS", "128");
        if let Some(def) = cc_def {
            nvcc.define(def, None);
        }
        if let Some(include) = env::var_os("DEP_SEMOLINA_C_INCLUDE") {
            nvcc.include(include);
        }
        if let Some(include) = env::var_os("DEP_SPPARK_ROOT") {
            nvcc.include(include);
        }
        nvcc.file("cuda/pallas.cu")
            .file("cuda/vesta.cu")
            .compile("pasta_msm_cuda");

        println!("cargo:rerun-if-changed=cuda");
        println!("cargo:rerun-if-env-changed=CXXFLAGS");
        println!("cargo:rustc-cfg=feature=\"cuda\"");
    }
    println!("cargo:rerun-if-env-changed=NVCC");
}
