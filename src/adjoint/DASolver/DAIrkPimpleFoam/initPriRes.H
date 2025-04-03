// Initialize primal residuals
volVectorField U1Res(
    IOobject(
        "U1Res",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedVector("U1Res", dimensionSet(0, 1, -2, 0, 0, 0, 0), vector::zero),
    zeroGradientFvPatchField<vector>::typeName);
volVectorField U2Res("U2Res", U1Res);

volScalarField p1Res(
    IOobject(
        "p1Res",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("p1Res", dimensionSet(0, 0, -1, 0, 0, 0, 0), 0.0),
    zeroGradientFvPatchField<scalar>::typeName);
volScalarField p2Res("p2Res", p1Res);

surfaceScalarField phi1Res(
    IOobject(
        "phi1Res",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    phi * 0.0);
surfaceScalarField phi2Res("phi2Res", phi1Res);

volScalarField nuTilda1Res(
    IOobject(
        "nuTilda1Res",
        runTime.timeName(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE),
    mesh,
    dimensionedScalar("nuTilda1Res", dimensionSet(0, 2, -2, 0, 0, 0, 0), 0.0),
    zeroGradientFvPatchField<scalar>::typeName);
volScalarField nuTilda2Res("nuTilda2Res", nuTilda1Res);
