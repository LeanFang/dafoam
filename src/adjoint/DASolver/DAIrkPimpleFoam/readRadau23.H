// Read states for Radau23-adjoint
// Read (U0, nuTilda0), (U1, p1, phi1, nuTilda1, nut1), and (U2, p2, phi2, nuTilda2, nut2) from disk

// ------ Read U0, nuTilda0 as oldTime U, nuTilda
{
    volVectorField readU0(
        IOobject(
            "U",
            Foam::name(timeInstance - deltaT),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    U = readU0;
}
U.correctBoundaryConditions();

{
    volScalarField readNuTilda0(
        IOobject(
            "nuTilda",
            Foam::name(timeInstance - deltaT),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    nuTilda = readNuTilda0;
}
nuTilda.correctBoundaryConditions();

// ------ Read U1, p1, phi1 as U1, p1, phi1
{
    volVectorField readU1(
        IOobject(
            "U1",
            Foam::name(timeInstance),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    U1 = readU1;
}
U1.correctBoundaryConditions();

{
    volScalarField readP1(
        IOobject(
            "p1",
            Foam::name(timeInstance),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    p1 = readP1;
}
p1.correctBoundaryConditions();

{
    surfaceScalarField readPhi1(
        IOobject(
            "phi1",
            Foam::name(timeInstance),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE),
        fvc::flux(U1));
    phi1 = readPhi1;
}

{
    volScalarField readNuTilda1(
        IOobject(
            "nuTilda1",
            Foam::name(timeInstance),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    nuTilda1 = readNuTilda1;
}
nuTilda1.correctBoundaryConditions();

{
    volScalarField readNut1(
        IOobject(
            "nut1",
            Foam::name(timeInstance),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    nut1 = readNut1;
}
nut1.correctBoundaryConditions();

// ------ Read U2, p2, phi2 as U, p, phi
{
    volVectorField readU2(
        IOobject(
            "U",
            Foam::name(timeInstance),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    U2 = readU2;
}
U2.correctBoundaryConditions();

{
    volScalarField readP2(
        IOobject(
            "p",
            Foam::name(timeInstance),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    p2 = readP2;
}
p2.correctBoundaryConditions();

{
    surfaceScalarField readPhi2(
        IOobject(
            "phi",
            Foam::name(timeInstance),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE),
        fvc::flux(U2));
    phi2 = readPhi2;
}

{
    volScalarField readNuTilda2(
        IOobject(
            "nuTilda",
            Foam::name(timeInstance),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    nuTilda2 = readNuTilda2;
}
nuTilda2.correctBoundaryConditions();

{
    volScalarField readNut2(
        IOobject(
            "nut",
            Foam::name(timeInstance),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE),
        mesh);
    nut2 = readNut2;
}
nut2.correctBoundaryConditions();
