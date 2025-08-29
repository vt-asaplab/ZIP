use std::{env, path::PathBuf};
use ark_bls12_381::{Bls12_381, Fr};
use ark_poly::{univariate::DensePolynomial, EvaluationDomain};
use ark_poly_commit::{Polynomial, UVPolynomial};
use ark_std::{test_rng, time::Instant, UniformRand};
use caulk::{
    multi::{
        compute_lookup_proof, get_poly_and_g2_openings, verify_lookup_proof, LookupInstance,
        LookupProverInput,
    },
    KZGCommit, PublicParameters,
};
use std::cmp::max;
use std::fs::{OpenOptions, create_dir_all};
use std::io::Write;

struct Config {
    n: usize,
    m: usize,
    positions: Vec<usize>,
    runs: usize,
}

fn times_file_path() -> PathBuf {
    // Allow override via env var if you want an absolute path
    if let Ok(p) = std::env::var("ZIP_TIMES_FILE") {
        return PathBuf::from(p);
    }
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../../proof_times.txt")
}

fn append_times(prove_secs: f64, verify_secs: f64) {
    let path = times_file_path();
    if let Some(parent) = path.parent() {
        let _ = create_dir_all(parent);
    }
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            let _ = writeln!(f, "{:.6}, {:.6}", prove_secs, verify_secs);
        }
        Err(e) => eprintln!("warn: cannot open {}: {}", path.display(), e),
    }
}

fn parse_cli() -> Config {
    use std::env;

    let args: Vec<String> = env::args().collect(); // [program, ...]
    let start = args.iter().position(|a| a == "--").map(|i| i + 1).unwrap_or(1);
    let mut it = args.iter().skip(start);

    let mut n: Option<usize> = None;
    let mut m: Option<usize> = None;
    let mut positions: Option<Vec<usize>> = None;
    let mut runs: usize = 1;

    while let Some(arg) = it.next() {
        match arg.as_str() {
            "--n" => {
                let v = it.next().expect("missing value for --n");
                n = Some(v.parse().expect("invalid --n"));
            }
            "--m" => {
                let v = it.next().expect("missing value for --m");
                m = Some(v.parse().expect("invalid --m"));
            }
            "--positions" => {
                let v = it.next().expect("missing value for --positions");
                let list = v
                    .split(',')
                    .filter(|s| !s.trim().is_empty())
                    .map(|s| s.trim().parse::<usize>().expect("invalid position"))
                    .collect::<Vec<_>>();
                positions = Some(list);
            }
            "--runs" => {
                let v = it.next().expect("missing value for --runs");
                runs = v.parse().expect("invalid --runs");
            }
            x => panic!("Unknown flag {x}. Expected --n --m --positions --runs"),
        }
    }

    let n = n.expect("required: --n <bits>");
    let m = m.expect("required: --m <count>");
    let positions = positions.unwrap_or_else(|| (0..m).collect());

    if positions.len() != m {
        panic!("--positions length ({}) must equal --m ({})", positions.len(), m);
    }

    Config { n, m, positions, runs }
}

#[allow(non_snake_case)]
fn main() {

    let cfg = parse_cli();
    let caulk_root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../caulk");
    env::set_current_dir(&caulk_root).expect("failed to chdir to ../../caulk");
    let mut rng = test_rng();

    // 1. Setup
    let n: usize = cfg.n;
    let m: usize = cfg.m;

    let N: usize = 1 << n;
    let powers_size: usize = max(N + 2, 1024);
    let actual_degree = N - 1;
    let temp_m = n; // dummy

    assert!(
        cfg.positions.iter().all(|&p| p < N),
        "all positions must be < N (= 2^n)"
    );

    let now = Instant::now();
    let mut pp = PublicParameters::<Bls12_381>::setup(&powers_size, &N, &temp_m, &n);
    println!(
        "Time to setup multi openings of table size {:?} = {:?}",
        actual_degree + 1,
        now.elapsed()
    );

    // 2. Poly and openings
    let now = Instant::now();
    let table = get_poly_and_g2_openings(&pp, actual_degree);
    println!("Time to generate commitment table = {:?}", now.elapsed());

    // 3. Setup
    pp.regenerate_lookup_params(m);

    // 4. Positions (from CLI)
    let positions: Vec<usize> = cfg.positions.clone();
    println!("positions = {:?}", positions);

    // 5. generating phi
    let blinder = Fr::rand(&mut rng);
    let a_m = DensePolynomial::from_coefficients_slice(&[blinder]);
    let mut phi_poly = a_m.mul_by_vanishing_poly(pp.domain_m);
    let c_poly_local = table.c_poly.clone();

    for j in 0..m {
        phi_poly = &phi_poly
            + &(&pp.lagrange_polynomials_m[j]
                * c_poly_local.evaluate(&pp.domain_N.element(positions[j]))); // adding c(w^{i_j})*mu_j(X)
    }

    for j in m..pp.domain_m.size() {
        phi_poly = &phi_poly
            + &(&pp.lagrange_polynomials_m[j] * c_poly_local.evaluate(&pp.domain_N.element(0)));
        // adding c(w^{i_j})*mu_j(X)
    }

    // 6. Running proofs
    let now = Instant::now();
    let c_com = KZGCommit::<Bls12_381>::commit_g1(&pp.poly_ck, &table.c_poly);
    let phi_com = KZGCommit::<Bls12_381>::commit_g1(&pp.poly_ck, &phi_poly);
    println!("Time to generate inputs = {:?}", now.elapsed());

    let lookup_instance = LookupInstance { c_com, phi_com };

    let prover_input = LookupProverInput {
        c_poly: table.c_poly.clone(),
        phi_poly,
        positions,
        openings: table.openings.clone(),
    };

    let number_of_openings: usize = cfg.runs;
    //println!("Running prover {} time(s)...", number_of_openings);    
    let now = Instant::now();
    let (proof, unity_proof) = compute_lookup_proof(&lookup_instance, &prover_input, &pp, &mut rng);
    for _ in 1..number_of_openings {
        _ = compute_lookup_proof(&lookup_instance, &prover_input, &pp, &mut rng);
    }
    let eval_elapsed = now.elapsed();
    println!(
        "Time to evaluate {} times {} multi-openings of table size {:?} = {:?} ",
        number_of_openings,
        m,
        N,
        eval_elapsed
    );

    let now = Instant::now();
    for _ in 0..number_of_openings {
        verify_lookup_proof(&table.c_com, &phi_com, &proof, &unity_proof, &pp, &mut rng);
    }
    let verify_elapsed = now.elapsed();
    println!(
        "Time to verify {} times {} multi-openings of table size {:?} = {:?} ",
        number_of_openings,
        m,
        N,
        verify_elapsed
    );

    assert!(
        verify_lookup_proof(&table.c_com, &phi_com, &proof, &unity_proof, &pp, &mut rng),
        "Result does not verify"
    );

    append_times(eval_elapsed.as_secs_f64(), verify_elapsed.as_secs_f64());

}

