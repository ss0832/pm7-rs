// SPDX-License-Identifier: GPL-3.0-or-later
//! Command-line interface for PM7 single points, gradients, optimization, and frequencies.

use pm7_rs::{
    closed_form_gradient, optimize, run_pm7, vibrational_analysis, Molecule, OptOptions,
    Pm7Options, Pm7Parameters, Pm7Method,
};
use std::process::exit;
use std::str::FromStr;

fn main() {
    if let Err(error) = run() {
        eprintln!("pm7-rs: {error}");
        exit(1);
    }
}

fn run() -> pm7_rs::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        print_usage();
        exit(2);
    }
    let mode = &args[1];
    let path = &args[2];
    let mut charge = 0.0;
    let mut multiplicity = 1usize;
    let mut method = Pm7Method::Pm7;
    let mut opt_output = None;
    let mut json = false;
    let mut use_diis = true;
    let mut exchange_cutoff: Option<(f64, f64)> = None;
    let mut index = 3;
    while index < args.len() {
        match args[index].as_str() {
            "--exchange-cutoff" => {
                // Two Bohr values: inner (full exchange) and outer (zero exchange), smooth between.
                let inner: f64 = parse(&args, index + 1, "--exchange-cutoff inner")?;
                let outer: f64 = parse(&args, index + 2, "--exchange-cutoff outer")?;
                exchange_cutoff = Some((inner, outer));
                index += 2;
            }
            "--charge" => {
                index += 1;
                charge = parse(&args, index, "--charge")?;
            }
            "--multiplicity" => {
                index += 1;
                multiplicity = parse(&args, index, "--multiplicity")?;
            }
            "--method" => {
                index += 1;
                method = Pm7Method::from_str(argument(&args, index, "--method")?)?;
            }
            "--opt-output" => {
                index += 1;
                opt_output = Some(argument(&args, index, "--opt-output")?.to_owned());
            }
            "--json" => json = true,
            "--no-diis" => use_diis = false,
            flag => {
                return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                    "unknown option `{flag}`"
                )))
            }
        }
        index += 1;
    }
    let molecule = Molecule::from_xyz_file(path, charge)?;
    let parameters = Pm7Parameters::method(method)?;
    let options = Pm7Options {
        method,
        charge,
        multiplicity,
        use_diis,
        exchange_cutoff,
        ..Pm7Options::default()
    };
    match mode.as_str() {
        "energy" | "charges" => {
            let result = run_pm7(&molecule, &parameters, &options)?;
            if json {
                println!("{{\"method\":\"{}\",\"energy_ev\":{:.12},\"heat_of_formation_kcal\":{:.12},\"charges\":[{}]}}", method, result.total_ev, result.heat_of_formation_kcal, result.charges.iter().map(|q| format!("{q:.12}")).collect::<Vec<_>>().join(","));
            } else {
                println!("method: {method}");
                println!("total energy: {:.12} eV", result.total_ev);
                println!(
                    "heat of formation: {:.12} kcal/mol",
                    result.heat_of_formation_kcal
                );
                println!("SCF iterations: {}", result.iterations);
                if mode == "charges" {
                    for (atom, charge) in molecule.atoms.iter().zip(&result.charges) {
                        println!(
                            "{} {charge:+.8}",
                            pm7_rs::z_to_symbol(atom.z).unwrap_or("X")
                        );
                    }
                }
            }
        }
        "gradient" => {
            let gradient = closed_form_gradient(&molecule, &parameters, &options)?;
            for vector in gradient.gradient {
                println!("{:.12} {:.12} {:.12}", vector.x, vector.y, vector.z);
            }
        }
        "optimize" => {
            let result = optimize(&molecule, &parameters, &options, &OptOptions::default())?;
            println!(
                "converged: {} after {} iterations",
                result.converged, result.iterations
            );
            println!("energy: {:.12} eV", result.scf.total_ev);
            if let Some(output) = opt_output {
                std::fs::write(output, xyz(&result.molecule, "pm7-rs optimized"))?;
            }
        }
        "frequencies" => {
            let modes = vibrational_analysis(&molecule, &parameters, &options, 1.0e-3)?;
            for frequency in modes.frequencies_cm {
                println!("{frequency:.6}");
            }
        }
        _ => {
            return Err(pm7_rs::Pm7Error::InvalidInput(format!(
                "unknown mode `{mode}`"
            )))
        }
    }
    Ok(())
}

fn argument<'a>(args: &'a [String], index: usize, flag: &str) -> pm7_rs::Result<&'a str> {
    args.get(index)
        .map(String::as_str)
        .ok_or_else(|| pm7_rs::Pm7Error::InvalidInput(format!("{flag} needs a value")))
}

fn parse<T: FromStr>(args: &[String], index: usize, flag: &str) -> pm7_rs::Result<T> {
    argument(args, index, flag)?
        .parse()
        .map_err(|_| pm7_rs::Pm7Error::InvalidInput(format!("invalid value for {flag}")))
}

fn xyz(molecule: &Molecule, title: &str) -> String {
    let mut text = format!("{}\n{title}\n", molecule.atoms.len());
    for atom in &molecule.atoms {
        text.push_str(&format!(
            "{} {:.12} {:.12} {:.12}\n",
            pm7_rs::z_to_symbol(atom.z).unwrap_or("X"),
            atom.position.x * pm7_rs::constants::BOHR_TO_ANGSTROM,
            atom.position.y * pm7_rs::constants::BOHR_TO_ANGSTROM,
            atom.position.z * pm7_rs::constants::BOHR_TO_ANGSTROM
        ));
    }
    text
}

fn print_usage() {
    eprintln!("usage: pm7_rs_cli <energy|gradient|charges|optimize|frequencies> structure.xyz [--method pm7] [--charge Q] [--multiplicity M] [--json] [--exchange-cutoff INNER OUTER]");
    eprintln!("  --exchange-cutoff INNER OUTER  smooth long-range-exchange cutoff (Bohr) for the analytic Hessian's CPHF; omitted = exact/bit-identical");
}
