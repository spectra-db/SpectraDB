//! TensorDB PostgreSQL Wire Protocol Server
//!
//! Accepts Postgres client connections (psql, JDBC, libpq, etc.) and routes
//! SQL queries to the embedded TensorDB engine.
//!
//! Usage:
//!   spectradb-server --data-dir ./data --port 5433

mod handler;
mod pgwire;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

use tokio::net::TcpListener;
use tracing::{error, info};

use spectradb_core::config::Config;
use spectradb_core::Database;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();
    let mut data_dir = PathBuf::from("./spectradb_data");
    let mut port: u16 = 5433;

    // Simple arg parsing
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--data-dir" | "-d" => {
                if i + 1 < args.len() {
                    data_dir = PathBuf::from(&args[i + 1]);
                    i += 2;
                } else {
                    eprintln!("--data-dir requires a path argument");
                    std::process::exit(1);
                }
            }
            "--port" | "-p" => {
                if i + 1 < args.len() {
                    port = args[i + 1].parse().unwrap_or_else(|_| {
                        eprintln!("invalid port number: {}", args[i + 1]);
                        std::process::exit(1);
                    });
                    i += 2;
                } else {
                    eprintln!("--port requires a number argument");
                    std::process::exit(1);
                }
            }
            "--help" | "-h" => {
                println!("tensordb-server - PostgreSQL wire protocol server for TensorDB");
                println!();
                println!("Usage: spectradb-server [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -d, --data-dir <PATH>   Data directory (default: ./spectradb_data)");
                println!("  -p, --port <PORT>       Listen port (default: 5433)");
                println!("  -h, --help              Show this help");
                std::process::exit(0);
            }
            _ => {
                eprintln!("unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
    }

    // Open the database
    info!("opening database at {}", data_dir.display());
    let db = match Database::open(&data_dir, Config::default()) {
        Ok(db) => Arc::new(db),
        Err(e) => {
            error!("failed to open database: {e}");
            std::process::exit(1);
        }
    };

    // Start TCP listener
    let addr = format!("0.0.0.0:{port}");
    let listener = match TcpListener::bind(&addr).await {
        Ok(l) => l,
        Err(e) => {
            error!("failed to bind to {addr}: {e}");
            std::process::exit(1);
        }
    };

    info!("TensorDB pgwire server listening on {addr}");
    info!("Connect with: psql -h localhost -p {port} -U spectra");

    let conn_counter = AtomicU32::new(1);

    loop {
        match listener.accept().await {
            Ok((stream, _addr)) => {
                let db = Arc::clone(&db);
                let conn_id = conn_counter.fetch_add(1, Ordering::Relaxed);
                tokio::spawn(async move {
                    handler::handle_connection(stream, db, conn_id).await;
                });
            }
            Err(e) => {
                error!("accept error: {e}");
            }
        }
    }
}
