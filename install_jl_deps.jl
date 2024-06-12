#!/usr/bin/env julia

using Pkg

# Path to the Manifest.toml file
manifest_file = "Manifest.toml"

# Activate project environment
Pkg.activate(".")

# Instantiate project environment based on Manifest.toml
Pkg.instantiate()

println("Julia dependencies installed successfully.")

