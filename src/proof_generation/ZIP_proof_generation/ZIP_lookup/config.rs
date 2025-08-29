use std::{
    fs,
    path::{Path, PathBuf},
};

#[derive(Clone, Debug)]
pub struct LookupConfig {
    pub table_dir: PathBuf,
    pub target_filename: String,
    pub pad_to_pow2: bool,
}

impl LookupConfig {
    pub fn load() -> Self {
        let table_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../ZIP_proof_generation/precomputed_lookup_tables_ieee754_hex/target_table");
        let target_filename = "target_lookup_table.txt".to_string();

        LookupConfig {
            table_dir,
            target_filename,
            pad_to_pow2: true,
        }
    }
}

pub fn parse_hex_file_to_fr<E: ark_ec::PairingEngine>(path: &Path) -> Vec<E::Fr> {
    let content =
        fs::read_to_string(path).expect(&format!("Could not read {}", path.display()));
    let mut out = Vec::new();

    for raw in content.split(|c: char| c.is_whitespace() || c == ',' || c == ';') {
        let t = raw.trim();
        if t.is_empty() { continue; }

        let hex = t.strip_prefix("0x")
            .or_else(|| t.strip_prefix("0X"))
            .unwrap_or(t);

        if hex.chars().all(|c| c.is_ascii_hexdigit()) {
            if let Ok(v) = u64::from_str_radix(hex, 16) {
                out.push(E::Fr::from(v));
            }
        }
    }

    assert!(
        !out.is_empty(),
        "No hex tokens were found in {}. Ensure it contains values like 0x3ff4000000000000",
        path.display()
    );
    out
}
