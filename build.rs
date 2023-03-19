use std::{env, fs, path::{Path, PathBuf}};

fn get_kind(path: &PathBuf) -> Result<shaderc::ShaderKind, ()> {
    let extension = path.extension().unwrap().to_str().unwrap();
    match extension {
        "vert" => Ok(shaderc::ShaderKind::Vertex),
        "frag" => Ok(shaderc::ShaderKind::Fragment),
        _ => Err(())
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/shaders/*.vert");
    println!("cargo:rerun-if-changed=src/shaders/*.frag");

    let in_dir = Path::new("src/shaders");
    let out_dir = String::from(env::var_os("OUT_DIR").unwrap().to_str().unwrap());
    let compiler = shaderc::Compiler::new().unwrap();

    for entry in fs::read_dir(in_dir).unwrap() {
        if let Some(path) = entry.ok().map(|e| e.path()) {
            let extension = path.extension().unwrap().to_str().unwrap();
            if path.is_file() && (extension.ends_with("vert") || extension.ends_with("frag")) {
                let shader = fs::read(&path).unwrap();
                let binary = compiler.compile_into_spirv(
                    std::str::from_utf8(&shader).unwrap(), 
                    get_kind(&path).unwrap(), 
                    path.file_name().unwrap().to_str().unwrap(), 
                    "main", 
                    None
                ).unwrap();

                let out_path = Path::new(&out_dir).join(format!("{}.spirv", path.file_name().unwrap().to_str().unwrap()));
                fs::write(out_path, binary.as_binary_u8()).unwrap();
            }
        }
    }
}