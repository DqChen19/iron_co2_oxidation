# Data and Code for:
**New empirical kinetics of iron oxidation by CO₂ applied to micrometeorites imply a CO₂-rich Archean atmosphere**

This repository provides the experimental datasets and numerical modeling codes associated with the study:

> Chen, D., et al. (2026).
> *New empirical kinetics of iron oxidation by CO₂ applied to micrometeorites imply a CO₂-rich Archean atmosphere.*
> Geochimica et Cosmochimica Acta, 416, 183–197.

## Repository Contents

The repository includes:

- `Reaction_rate&emissivity.xlsx`: Experimental measurements of Fe–CO₂ reaction kinetics and thermal emissivity.
- **Micrometeorite entry model**: A revised one-dimensional (1D) atmospheric entry model incorporating experimentally constrained oxidation kinetics and atmospheric compositions derived from photochemical simulations.

## Main Results

### 1. Atmospheric Entry Model and Oxidation Mechanisms

We developed an improved 1D atmospheric entry model for iron-type cosmic spherules (ICSs), building upon the framework of [Lehmer et al. (2020)](https://www.science.org/doi/10.1126/sciadv.aay4644).

The model is used to investigate the atmospheric CO₂ abundance at approximately 2.7 Ga using fossil iron-type cosmic spherules.

Two major oxidation mechanisms are incorporated:

1. **Direct oxidation by atmospheric CO₂**, constrained by experimentally measured Fe–CO₂ reaction kinetics.
2. **Oxidation by photochemically produced atomic oxygen**, with atmospheric O abundances calculated using a photochemical model.

<p align="center">
  <img src="mechanism.png" width="700">
</p>

<p align="center">
  <em>
    Figure 1. Schematic illustration of the two oxidation pathways considered in the model:
    (a) direct oxidation of metallic Fe by atmospheric CO₂;
    (b) oxidation of metallic Fe by atomic oxygen produced through atmospheric photochemistry.
  </em>
</p>

### 2. Experimental Fe–CO₂ Reaction Kinetics

We experimentally measured the oxidation kinetics of metallic iron by CO₂ under a range of temperatures and CO₂ partial pressures.

These measurements provide empirical constraints on the temperature- and pressure-dependent reaction rates used in the atmospheric entry model.

<p align="center">
  <img src="rates.png" width="700">
</p>

<p align="center">
  <em>
    Figure 2. Experimentally measured oxidation rates of metallic Fe by CO₂ at temperatures below 1650 K.
  </em>
</p>

### 3. Micrometeorite Atmospheric Entry Simulations

By incorporating the experimentally determined oxidation kinetics and photochemically modeled atmospheric compositions, the revised entry model simulates the coupled thermal, dynamical, and chemical evolution of iron micrometeorites during atmospheric entry.
<p align="center">
  <img src="single.png" width="700">
</p>

<p align="center">
  <em>
    Figure 3. Micrometeorite trajectories.
  </em>
</p>
