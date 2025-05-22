use std::process::{Command, Stdio};

use anyhow::{anyhow, Context, Result};
use dialoguer::{theme::ColorfulTheme, Confirm, MultiSelect};

fn main() -> Result<()> {
    // 1️⃣  Discover services -------------------------------------------------
    let services = discover_services()?;
    if services.is_empty() {
        eprintln!(
            "\n❗  No services found in the current directory.\n   Are you inside a folder that contains a docker‑compose.yml / compose.yaml?\n   Try running   docker compose config --services   manually to verify.\n"
        );
        std::process::exit(1);
    }

    // 2️⃣  Multi‑select prompt ---------------------------------------------
    let selections = MultiSelect::with_theme(&ColorfulTheme::default())
        .with_prompt("Select services to start (Space = toggle, Enter = done)")
        .items(&services)
        .interact()?;

    // 3️⃣  Flag prompts ------------------------------------------------------
    let detached = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Run in detached mode (‑d)?")
        .default(true)
        .interact()?;

    let rebuild = Confirm::with_theme(&ColorfulTheme::default())
        .with_prompt("Rebuild images even if they exist (--build)?")
        .default(false)
        .interact()?;

    // 4️⃣  Assemble docker compose command ----------------------------------
    let mut args: Vec<String> = vec!["compose".into(), "up".into()];
    if detached {
        args.push("-d".into());
    }
    if rebuild {
        args.push("--build".into());
    }

    // Handle zero‑selection case gracefully
    if selections.is_empty() {
        // Offer to bring up *all* services or cancel
        let run_all = Confirm::with_theme(&ColorfulTheme::default())
            .with_prompt("No services selected – start ALL services?")
            .default(true)
            .interact()?;
        if !run_all {
            println!("Aborted – nothing started.");
            return Ok(());
        }
        args.extend(services.clone());
    } else {
        for idx in selections {
            args.push(services[idx].clone());
        }
    }

    println!("\n> docker {}\n", args.join(" "));

    // 5️⃣  Spawn and stream output ------------------------------------------
    let status = Command::new("docker")
        .args(&args)
        .stdin(Stdio::inherit())
        .stdout(Stdio::inherit())
        .stderr(Stdio::inherit())
        .status()
        .context("failed to launch docker")?;

    std::process::exit(status.code().unwrap_or(1));
}

// Helper — run `docker compose config --services` and split lines -------------
fn discover_services() -> Result<Vec<String>> {
    let output = Command::new("docker")
        .args(["compose", "config", "--services"])
        .output()
        .context("running 'docker compose config --services'")?;

    if !output.status.success() {
        // Pass Docker’s stderr through so the user sees the root cause.
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!(
            "docker compose reported an error:\n{}",
            stderr.trim_end()
        ));
    }

    let stdout = String::from_utf8(output.stdout)?;
    Ok(stdout
        .lines()
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect())
}
