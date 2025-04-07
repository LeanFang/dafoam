// Construct the lhs pseudo equations
// Note that we only care about the lhs

// * * * * * * * * * * * * * * * * * * //
// 1st stage

// ------ pseudo_U1Eqn -----
// Duplicate U1 as pseudo_U1
volVectorField pseudo_U1("pseudo_U1", U1);

// Get nuEff1
this->correctNut(nut1, nuTilda1, nu);
volScalarField nuEff1("nuEff1", nu + nut1);

// Initialize pseudo_U1Eqn w/o ddt term
fvVectorMatrix pseudo_U1Eqn(
    fvm::div(phi1, pseudo_U1, divUScheme)
    //+ turbulence->divDevReff(U)
    - fvm::laplacian(nuEff1, pseudo_U1));

// Update pseudo_U1Eqn with pseudo-spectral terms
forAll(pseudo_U1, cellI)
{
    scalar meshV = pseudo_U1.mesh().V()[cellI];

    // Add D11 / halfDeltaT[i] * V() to diagonal
    pseudo_U1Eqn.diag()[cellI] += D11 / deltaT * meshV;
}

// We use relaxation factor = 1.0, cannot skip the step below
pseudo_U1Eqn.relax(relaxUEqn);

// swap upper() and lower()
this->swap(pseudo_U1Eqn.upper(), pseudo_U1Eqn.lower());

// Get rAU
volScalarField rAU1(1.0 / pseudo_U1Eqn.A());
tmp<volScalarField> rAtU1(rAU1);

// ------ pseudo_p1Eqn -----
// Duplicate p1 as pseudo_p1
volScalarField pseudo_p1("pseudo_p1", p1);

// Get the pseudo_p1Eqn
fvScalarMatrix pseudo_p1Eqn(
    fvm::laplacian(rAtU1(), pseudo_p1));

// swap upper() and lower()
this->swap(pseudo_p1Eqn.upper(), pseudo_p1Eqn.lower());

// ------ pseudo_nuTilda1Eqn -----
// Duplicate nuTilda1 as pseudo_nuTilda1
volScalarField pseudo_nuTilda1("pseudo_nuTilda1", nuTilda1);

// Get chi1 and fv11
volScalarField chi1("chi1", this->chi(pseudo_nuTilda1, nu));
volScalarField fv11("fv11", this->fv1(chi1));

// Get Stilda1
volScalarField Stilda1(
    "Stilda1",
    this->fv3(chi1, fv11) * ::sqrt(2.0) * mag(skew(fvc::grad(U1))) + this->fv2(chi1, fv11) * pseudo_nuTilda1 / sqr(kappa * y));

// Construct pseudo_nuTilda1Eqn w/o ddt term
fvScalarMatrix pseudo_nuTilda1Eqn(
    fvm::div(phi1, pseudo_nuTilda1, divNuTildaScheme)
        - fvm::laplacian(DnuTildaEff(pseudo_nuTilda1, nu), pseudo_nuTilda1)
    == -fvm::Sp(Cw1 * this->fw(Stilda1, pseudo_nuTilda1, y) * pseudo_nuTilda1 / sqr(y), pseudo_nuTilda1));

// Update pseudo_nuTilda1Eqn with pseudo-spectral terms
forAll(pseudo_nuTilda1, cellI)
{
    scalar meshV = pseudo_nuTilda1.mesh().V()[cellI];

    // Add D11 / halfDeltaT[i] * V() to diagonal
    pseudo_nuTilda1Eqn.diag()[cellI] += D11 / deltaT * meshV;
}

pseudo_nuTilda1Eqn.relax(relaxNuTildaEqn);

// swap upper() and lower()
this->swap(pseudo_nuTilda1Eqn.upper(), pseudo_nuTilda1Eqn.lower());

// * * * * * * * * * * * * * * * * * * //
// 2nd stage

// ------ pseudo_U2Eqn -----
// Duplicate U2 as pseudo_U2
volVectorField pseudo_U2("pseudo_U2", U2);

// Get nuEff2
this->correctNut(nut2, nuTilda2, nu);
volScalarField nuEff2("nuEff2", nu + nut2);

// Initialize pseudo_U2Eqn w/o ddt term
fvVectorMatrix pseudo_U2Eqn(
    fvm::div(phi2, pseudo_U2, divUScheme)
    //+ turbulence->divDevReff(U)
    - fvm::laplacian(nuEff2, pseudo_U2));

// Update pseudo_U2Eqn with pseudo-spectral terms
forAll(pseudo_U2, cellI)
{
    scalar meshV = pseudo_U2.mesh().V()[cellI];

    // Add D22 / halfDeltaT[i] * V() to diagonal
    pseudo_U2Eqn.diag()[cellI] += D22 / deltaT * meshV;
}

// We use relaxation factor = 1.0, cannot skip the step below
pseudo_U2Eqn.relax(relaxUEqn);

// swap upper() and lower()
this->swap(pseudo_U2Eqn.upper(), pseudo_U2Eqn.lower());

// Get rAU
volScalarField rAU2(1.0 / pseudo_U2Eqn.A());
tmp<volScalarField> rAtU2(rAU2);

// ------ pseudo_p2Eqn -----
// Duplicate p2 as pseudo_p2
volScalarField pseudo_p2("pseudo_p2", p2);

// Get the pseudo_p2Eqn
fvScalarMatrix pseudo_p2Eqn(
    fvm::laplacian(rAtU2(), pseudo_p2));

// swap upper() and lower()
this->swap(pseudo_p2Eqn.upper(), pseudo_p2Eqn.lower());

// ------ pseudo_nuTilda2Eqn -----
// Duplicate nuTilda2 as pseudo_nuTilda2
volScalarField pseudo_nuTilda2("pseudo_nuTilda2", nuTilda2);

// Get chi2 and fv12
volScalarField chi2("chi2", this->chi(pseudo_nuTilda2, nu));
volScalarField fv12("fv12", this->fv1(chi2));

// Get Stilda2
volScalarField Stilda2(
    "Stilda2",
    this->fv3(chi2, fv12) * ::sqrt(2.0) * mag(skew(fvc::grad(U2))) + this->fv2(chi2, fv12) * pseudo_nuTilda2 / sqr(kappa * y));

// Construct pseudo_nuTilda2Eqn w/o ddt term
fvScalarMatrix pseudo_nuTilda2Eqn(
    fvm::div(phi2, pseudo_nuTilda2, divNuTildaScheme)
        - fvm::laplacian(DnuTildaEff(pseudo_nuTilda2, nu), pseudo_nuTilda2)
    == -fvm::Sp(Cw1 * this->fw(Stilda2, pseudo_nuTilda2, y) * pseudo_nuTilda2 / sqr(y), pseudo_nuTilda2));

// Update pseudo_nuTilda2Eqn with pseudo-spectral terms
forAll(pseudo_nuTilda2, cellI)
{
    scalar meshV = pseudo_nuTilda2.mesh().V()[cellI];

    // Add D22 / halfDeltaT[i] * V() to diagonal
    pseudo_nuTilda2Eqn.diag()[cellI] += D22 / deltaT * meshV;
}

pseudo_nuTilda2Eqn.relax(relaxNuTildaEqn);

// swap upper() and lower()
this->swap(pseudo_nuTilda2Eqn.upper(), pseudo_nuTilda2Eqn.lower());
