/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    This class is modified from OpenFOAM's source code
    applications/solvers/incompressible/pimpleFoam

    OpenFOAM: The Open Source CFD Toolbox

    Copyright (C): 2011-2016 OpenFOAM Foundation

    OpenFOAM License:

        OpenFOAM is free software: you can redistribute it and/or modify it
        under the terms of the GNU General Public License as published by
        the Free Software Foundation, either version 3 of the License, or
        (at your option) any later version.
    
        OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
        ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
        FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
        for more details.
    
        You should have received a copy of the GNU General Public License
        along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "DAIrkPimpleFoam.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

defineTypeNameAndDebug(DAIrkPimpleFoam, 0);
addToRunTimeSelectionTable(DASolver, DAIrkPimpleFoam, dictionary);
// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

DAIrkPimpleFoam::DAIrkPimpleFoam(
    char* argsAll,
    PyObject* pyOptions)
    : DASolver(argsAll, pyOptions),
      // Radau23 coefficients and weights
      D10(-2),
      D11(3.0 / 2),
      D12(1.0 / 2),
      D20(2),
      D21(-9.0 / 2),
      D22(5.0 / 2),
      w1(3.0 / 4),
      w2(1.0 / 4),

      // SA-fv3 model coefficients
      sigmaNut(0.66666),
      kappa(0.41),
      Cb1(0.1355),
      Cb2(0.622),
      Cw1(Cb1 / sqr(kappa) + (1.0 + Cb2) / sigmaNut),
      Cw2(0.3),
      Cw3(2.0),
      Cv1(7.1),
      Cv2(5.0)
{
}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Functions for SA-fv3 model
#include "mySAModel.H"

// Some utilities, move to DAUtility later
#include "myUtilities.H"

void DAIrkPimpleFoam::calcPriResIrkOrig(
    volVectorField& U0, // oldTime U
    volVectorField& U1, // 1st stage
    volScalarField& p1,
    surfaceScalarField& phi1,
    volScalarField& nuTilda1,
    volScalarField& nut1,
    volVectorField& U2, // 2nd stage
    volScalarField& p2,
    surfaceScalarField& phi2,
    volScalarField& nuTilda2,
    volScalarField& nut2,
    const volScalarField& nu,
    const scalar& deltaT, // current dt
    volVectorField& U1Res, // Residual for 1st stage
    volScalarField& p1Res,
    surfaceScalarField& phi1Res,
    volVectorField& U2Res, // Residual for end stage
    volScalarField& p2Res,
    surfaceScalarField& phi2Res,
    const scalar& relaxUEqn)
{
    // Numerical settings
    word divUScheme = "div(phi,U)";
    word divGradUScheme = "div((nuEff*dev2(T(grad(U)))))";

    // Update boundaries
    U0.correctBoundaryConditions(); // oldTime U
    U1.correctBoundaryConditions(); // 1st stage
    p1.correctBoundaryConditions();
    nuTilda1.correctBoundaryConditions();
    U2.correctBoundaryConditions(); // 2nd stage
    p2.correctBoundaryConditions();
    nuTilda2.correctBoundaryConditions();

    // --- 1st stage
    {
        // Get nuEff1
        this->correctNut(nut1, nuTilda1, nu);
        volScalarField nuEff1("nuEff1", nu + nut1);

        // Initialize U1Eqn w/o ddt term
        fvVectorMatrix U1Eqn(
            fvm::div(phi1, U1, divUScheme)
            //+ turbulence->divDevReff(U)
            - fvm::laplacian(nuEff1, U1)
            - fvc::div(nuEff1 * dev2(T(fvc::grad(U1))), divGradUScheme));

        // Update U1Eqn with pseudo-spectral terms
        forAll(U1, cellI)
        {
            scalar meshV = U1.mesh().V()[cellI];

            // Add D11 / halfDeltaT[i] * V() to diagonal
            U1Eqn.diag()[cellI] += D11 / deltaT * meshV; // Use one seg for now: halfDeltaTList[segI]

            // Minus D10 / halfDeltaT[i] * T0 * V() to source term
            U1Eqn.source()[cellI] -= D10 / deltaT * U0[cellI] * meshV;

            // Minus D12 / halfDeltaT[i] * T2 * V() to source term
            U1Eqn.source()[cellI] -= D12 / deltaT * U2[cellI] * meshV;
        }

        U1Res = (U1Eqn & U1) + fvc::grad(p1);

        // We use relaxation factor = 1.0, cannot skip the step below
        U1Eqn.relax(relaxUEqn);

        volScalarField rAU1(1.0 / U1Eqn.A());
        volVectorField HbyA1(constrainHbyA(rAU1 * U1Eqn.H(), U1, p1));
        surfaceScalarField phiHbyA1(
            "phiHbyA1",
            fvc::flux(HbyA1));
        tmp<volScalarField> rAtU1(rAU1);

        fvScalarMatrix p1Eqn(
            fvm::laplacian(rAtU1(), p1) == fvc::div(phiHbyA1));

        p1Res = p1Eqn & p1;

        // Then do phiRes
        phi1Res = phiHbyA1 - p1Eqn.flux() - phi1;
    }

    // --- 2nd stage
    {
        // Get nuEff2
        this->correctNut(nut2, nuTilda2, nu);
        volScalarField nuEff2("nuEff2", nu + nut2);

        // Initialize U2Eqn w/o ddt term
        fvVectorMatrix U2Eqn(
            fvm::div(phi2, U2, divUScheme)
            //+ turbulence->divDevReff(U)
            - fvm::laplacian(nuEff2, U2)
            - fvc::div(nuEff2 * dev2(T(fvc::grad(U2))), divGradUScheme));

        // Update U2Eqn with pseudo-spectral terms
        forAll(U2, cellI)
        {
            scalar meshV = U2.mesh().V()[cellI];

            // Add D22 / halfDeltaT[i] * V() to diagonal
            U2Eqn.diag()[cellI] += D22 / deltaT * meshV; // Use one seg for now: halfDeltaTList[segI]

            // Minus D20 / halfDeltaT[i] * T0 * V() to source term
            U2Eqn.source()[cellI] -= D20 / deltaT * U0[cellI] * meshV;

            // Minus D21 / halfDeltaT[i] * T2 * V() to source term
            U2Eqn.source()[cellI] -= D21 / deltaT * U1[cellI] * meshV;
        }

        U2Res = (U2Eqn & U2) + fvc::grad(p2);

        // We use relaxation factor = 1.0, cannot skip the step below
        U2Eqn.relax(relaxUEqn);

        volScalarField rAU2(1.0 / U2Eqn.A());
        volVectorField HbyA2(constrainHbyA(rAU2 * U2Eqn.H(), U2, p2));
        surfaceScalarField phiHbyA2(
            "phiHbyA2",
            fvc::flux(HbyA2));
        tmp<volScalarField> rAtU2(rAU2);

        // Non-orthogonal pressure corrector loop
        fvScalarMatrix p2Eqn(
            fvm::laplacian(rAtU2(), p2) == fvc::div(phiHbyA2));

        p2Res = p2Eqn & p2;

        // Then do phiRes
        phi2Res = phiHbyA2 - p2Eqn.flux() - phi2;
    }
}

void DAIrkPimpleFoam::calcPriSAResIrkOrig(
    volScalarField& nuTilda0, // oldTime nuTilda
    volVectorField& U1, // 1st stage
    surfaceScalarField& phi1,
    volScalarField& nuTilda1,
    volVectorField& U2, // 2nd stage
    surfaceScalarField& phi2,
    volScalarField& nuTilda2,
    volScalarField& y,
    const volScalarField& nu,
    const scalar& deltaT, // current dt
    volScalarField& nuTilda1Res, // Residual for 1st stage
    volScalarField& nuTilda2Res) // Residual for 2nd stage
{
    // Numerical settings
    word divNuTildaScheme = "div(phi,nuTilda)";

    // Update boundaries
    nuTilda0.correctBoundaryConditions();
    U1.correctBoundaryConditions();
    nuTilda1.correctBoundaryConditions();
    U2.correctBoundaryConditions();
    nuTilda2.correctBoundaryConditions();

    // --- 1st stage
    {
        // Get chi1 and fv11
        volScalarField chi1("chi1", this->chi(nuTilda1, nu));
        volScalarField fv11("fv11", this->fv1(chi1));

        // Get Stilda1
        volScalarField Stilda1(
            "Stilda1",
            this->fv3(chi1, fv11) * ::sqrt(2.0) * mag(skew(fvc::grad(U1))) + this->fv2(chi1, fv11) * nuTilda1 / sqr(kappa * y));

        // Construct nuTilda1Eqn w/o ddt term
        fvScalarMatrix nuTilda1Eqn(
            fvm::div(phi1, nuTilda1, divNuTildaScheme)
                - fvm::laplacian(DnuTildaEff(nuTilda1, nu), nuTilda1)
                - Cb2 / sigmaNut * magSqr(fvc::grad(nuTilda1))
            == Cb1 * Stilda1 * nuTilda1 // If field inversion, beta should be multiplied here
                - fvm::Sp(Cw1 * this->fw(Stilda1, nuTilda1, y) * nuTilda1 / sqr(y), nuTilda1));

        // Update nuTilda1Eqn with pseudo-spectral terms
        forAll(nuTilda1, cellI)
        {
            scalar meshV = nuTilda1.mesh().V()[cellI];

            // Add D11 / halfDeltaT[i] * V() to diagonal
            nuTilda1Eqn.diag()[cellI] += D11 / deltaT * meshV;

            // Minus D10 / halfDeltaT[i] * T0 * V() to source term
            nuTilda1Eqn.source()[cellI] -= D10 / deltaT * nuTilda0[cellI] * meshV;

            // Minus D12 / halfDeltaT[i] * T2 * V() to source term
            nuTilda1Eqn.source()[cellI] -= D12 / deltaT * nuTilda2[cellI] * meshV;
        }

        nuTilda1Res = nuTilda1Eqn & nuTilda1;
    }

    // --- 2nd stage
    {
        // Get chi2 and fv12
        volScalarField chi2("chi2", this->chi(nuTilda2, nu));
        volScalarField fv12("fv12", this->fv1(chi2));

        // Get Stilda2
        volScalarField Stilda2(
            "Stilda2",
            this->fv3(chi2, fv12) * ::sqrt(2.0) * mag(skew(fvc::grad(U2))) + this->fv2(chi2, fv12) * nuTilda2 / sqr(kappa * y));

        // Construct nuTilda2Eqn w/o ddt term
        fvScalarMatrix nuTilda2Eqn(
            fvm::div(phi2, nuTilda2, divNuTildaScheme)
                - fvm::laplacian(DnuTildaEff(nuTilda2, nu), nuTilda2)
                - Cb2 / sigmaNut * magSqr(fvc::grad(nuTilda2))
            == Cb1 * Stilda2 * nuTilda2 // If field inversion, beta should be multiplied here
                - fvm::Sp(Cw1 * this->fw(Stilda2, nuTilda2, y) * nuTilda2 / sqr(y), nuTilda2));

        // Update nuTilda2Eqn with pseudo-spectral terms
        forAll(nuTilda2, cellI)
        {
            scalar meshV = nuTilda2.mesh().V()[cellI];

            // Add D22 / halfDeltaT[i] * V() to diagonal
            nuTilda2Eqn.diag()[cellI] += D22 / deltaT * meshV;

            // Minus D20 / halfDeltaT[i] * T0 * V() to source term
            nuTilda2Eqn.source()[cellI] -= D20 / deltaT * nuTilda0[cellI] * meshV;

            // Minus D21 / halfDeltaT[i] * T2 * V() to source term
            nuTilda2Eqn.source()[cellI] -= D21 / deltaT * nuTilda1[cellI] * meshV;
        }

        nuTilda2Res = nuTilda2Eqn & nuTilda2;
    }
}

// For FP adjoint, we need primal residual scaled with mesh volumes
// phiRes is already scaled with surface areas
void DAIrkPimpleFoam::scaledPriResIrk(
    volVectorField& U0, // oldTime U
    volScalarField& nuTilda0, // oldTime nuTilda
    volVectorField& U1, // 1st stage
    volScalarField& p1,
    surfaceScalarField& phi1,
    volScalarField& nuTilda1,
    volScalarField& nut1,
    volVectorField& U2, // 2nd stage
    volScalarField& p2,
    surfaceScalarField& phi2,
    volScalarField& nuTilda2,
    volScalarField& nut2,
    volScalarField& y,
    const volScalarField& nu,
    const scalar& deltaT, // current dt
    volVectorField& U1Res, // Residual for 1st stage
    volScalarField& p1Res,
    surfaceScalarField& phi1Res,
    volScalarField& nuTilda1Res,
    volVectorField& U2Res, // Residual for end stage
    volScalarField& p2Res,
    surfaceScalarField& phi2Res,
    volScalarField& nuTilda2Res,
    const scalar& relaxUEqn)
{
    this->calcPriResIrkOrig(U0, U1, p1, phi1, nuTilda1, nut1, U2, p2, phi2, nuTilda2, nut2, nu, deltaT, U1Res, p1Res, phi1Res, U2Res, p2Res, phi2Res, relaxUEqn);
    this->calcPriSAResIrkOrig(nuTilda0, U1, phi1, nuTilda1, U2, phi2, nuTilda2, y, nu, deltaT, nuTilda1Res, nuTilda2Res);

    forAll(U1Res, cellI)
    {
        scalar meshV = U1.mesh().V()[cellI];
        U1Res[cellI] = U1Res[cellI] * meshV;
    }
    forAll(p1Res, cellI)
    {
        scalar meshV = p1.mesh().V()[cellI];
        p1Res[cellI] = p1Res[cellI] * meshV;
    }
    forAll(nuTilda1Res, cellI)
    {
        scalar meshV = nuTilda1.mesh().V()[cellI];
        nuTilda1Res[cellI] = nuTilda1Res[cellI] * meshV;
    }

    forAll(U2Res, cellI)
    {
        scalar meshV = U2.mesh().V()[cellI];
        U2Res[cellI] = U2Res[cellI] * meshV;
    }
    forAll(p2Res, cellI)
    {
        scalar meshV = p2.mesh().V()[cellI];
        p2Res[cellI] = p2Res[cellI] * meshV;
    }
    forAll(nuTilda2Res, cellI)
    {
        scalar meshV = nuTilda2.mesh().V()[cellI];
        nuTilda2Res[cellI] = nuTilda2Res[cellI] * meshV;
    }
}

// We use meanU as the test objective function
scalar DAIrkPimpleFoam::calcMeanU(volVectorField& U)
{
    scalar meanU = 0.0;
    forAll(U, cellI)
    {
        for (label i = 0; i < 3; i++)
        {
            meanU += U[cellI][i];
        }
    }
    meanU /= U.size() * 3;
    return meanU;
}

void DAIrkPimpleFoam::initSolver()
{
    /*
    Description:
        Initialize variables for DASolver
    */
    daOptionPtr_.reset(new DAOption(meshPtr_(), pyOptions_));
}

label DAIrkPimpleFoam::solvePrimal()
{
    /*
    Description:
        Call the primal solver to get converged state variables

    Output:
        state variable vector
    */

    Foam::argList& args = argsPtr_();

#include "createTime.H"
#include "createMesh.H"
#include "initContinuityErrs.H"
#include "createControl.H"
#include "createFieldsIrkPimple.H"
#include "CourantNo.H"

    // Turbulence disabled
    //turbulence->validate();

    // Get nu, nut, nuTilda, y
    volScalarField& nu = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nu"));
    volScalarField& nut = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nut"));
    volScalarField& nuTilda = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nuTilda"));
    volScalarField& y = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("yWall"));

    // * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

    Info << "\nStarting time loop\n"
         << endl;

#include "IrkControl.H"

    // Duplicate state variables for stages
    volVectorField U1("U1", U);
    volVectorField U2("U2", U);
    volScalarField p1("p1", p);
    volScalarField p2("p2", p);
    surfaceScalarField phi1("phi1", phi);
    surfaceScalarField phi2("phi2", phi);
    // SA turbulence model
    volScalarField nuTilda1("nuTilda1", nuTilda);
    volScalarField nuTilda2("nuTilda2", nuTilda);
    volScalarField nut1("nut1", nut);
    volScalarField nut2("nut2", nut);

    // Settings for stage pressure
    mesh.setFluxRequired(p1.name());
    mesh.setFluxRequired(p2.name());

// Initialize primal residuals
#include "initPriRes.H"

    // Initialize oldTime() for under-relaxation
    U1.oldTime() = U1;
    U2.oldTime() = U2;
    p1.oldTime() = p1;
    p2.oldTime() = p2;
    phi1.oldTime() = phi1;
    phi2.oldTime() = phi2;
    nuTilda1.oldTime() = nuTilda1;
    nuTilda2.oldTime() = nuTilda2;

    // Numerical settings
    word divUScheme = "div(phi,U)";
    word divGradUScheme = "div((nuEff*dev2(T(grad(U)))))";
    word divNuTildaScheme = "div(phi,nuTilda)";

    const fvSolution& myFvSolution = mesh.thisDb().lookupObject<fvSolution>("fvSolution");
    dictionary solverDictU = myFvSolution.subDict("solvers").subDict("U");
    dictionary solverDictP = myFvSolution.subDict("solvers").subDict("p");
    dictionary solverDictNuTilda = myFvSolution.subDict("solvers").subDict("nuTilda");

    scalar endTime = runTime.endTime().value();

    // Initialize the obj timeAvgMeanU
    scalar timeAvgMeanU = 0.0;

    while (runTime.run())
    {

#include "CourantNo.H"

        ++runTime;

        Info << "Time = " << runTime.timeName() << nl << endl;

        scalar deltaT = runTime.deltaTValue();

        // --- GS sweeps for IRK-PIMPLE
        label sweepIndex = 0;
        while (sweepIndex < maxSweep)
        {
            Info << "Block GS sweep = " << sweepIndex + 1 << endl;

            // --- 1st stage
            {
#include "U1EqnIrkPimple.H"

                while (pimple.correct())
                {
#include "p1EqnIrkPimple.H"
                }

                // --- Correct turbulence, using our own SAFv3
#include "nuTilda1EqnIrkPimple.H"
            }

            // --- 2nd stage
            {
#include "U2EqnIrkPimple.H"

                while (pimple.correct())
                {
#include "p2EqnIrkPimple.H"
                }

                // --- Correct turbulence, using our own SAFv3
#include "nuTilda2EqnIrkPimple.H"
            }

            if (checkPriRes == "yes")
            {
                this->calcPriResIrkOrig(U, U1, p1, phi1, nuTilda1, nut1, U2, p2, phi2, nuTilda2, nut2, nu, deltaT, U1Res, p1Res, phi1Res, U2Res, p2Res, phi2Res, relaxUEqn);
                this->calcPriSAResIrkOrig(nuTilda, U1, phi1, nuTilda1, U2, phi2, nuTilda2, y, nu, deltaT, nuTilda1Res, nuTilda2Res);

                //this->scaledPriResIrk(U, nuTilda, U1, p1, phi1, nuTilda1, nut1, U2, p2, phi2, nuTilda2, nut2, y, nu, deltaT, U1Res, p1Res, phi1Res, nuTilda1Res, U2Res, p2Res, phi2Res, nuTilda2Res, relaxUEqn);

                Info << "L2 norm of U1Res: " << this->L2norm(U1Res.primitiveField()) << endl;
                Info << "L2 norm of U2Res: " << this->L2norm(U2Res.primitiveField()) << endl;
                Info << "L2 norm of p1Res: " << this->L2norm(p1Res.primitiveField()) << endl;
                Info << "L2 norm of p2Res: " << this->L2norm(p2Res.primitiveField()) << endl;
                Info << "L2 norm of phi1Res: " << this->L2norm(phi1Res, phi1.mesh().magSf()) << endl;
                Info << "L2 norm of phi2Res: " << this->L2norm(phi2Res, phi2.mesh().magSf()) << endl;
                Info << "L2 norm of nuTilda1Res: " << this->L2norm(nuTilda1Res.primitiveField()) << endl;
                Info << "L2 norm of nuTilda2Res: " << this->L2norm(nuTilda2Res.primitiveField()) << endl;
            }

            sweepIndex++;
        }

        // Update new step values before write-to-disk
        U = U2;
        U.correctBoundaryConditions();
        p = p2;
        p.correctBoundaryConditions();
        phi = phi2;
        nuTilda = nuTilda2;
        nuTilda.correctBoundaryConditions();
        nut = nut2;
        nut.correctBoundaryConditions();

        // Write to disk
        //runTime.write(); // This writes U, p, phi, nuTilda, nut
        U.write();
        p.write();
        phi.write();
        nuTilda.write();
        nut.write();
        // Also write internal stages to disk (Radau23)
        U1.write();
        p1.write();
        phi1.write();
        nuTilda1.write();
        nut1.write();

        // Calculate local obj: meanT1 and meanT2
        scalar meanU1 = calcMeanU(U1);
        scalar meanU2 = calcMeanU(U2);

        Info << "Local obj meanU1: " << meanU1 << endl;
        Info << "Local obj meanU2: " << meanU2 << endl;

        // Add to global obj
        timeAvgMeanU += meanU1 * w1 * deltaT / endTime;
        timeAvgMeanU += meanU2 * w2 * deltaT / endTime;

        // Use old step as initial guess for the next step
        U1 = U;
        U1.correctBoundaryConditions();
        p1 = p;
        p1.correctBoundaryConditions();
        phi1 = phi;
        nuTilda1 = nuTilda;
        nuTilda1.correctBoundaryConditions();
        nut1 = nut;
        nut1.correctBoundaryConditions();

        runTime.printExecutionTime(Info);
    }

    Info << "Global obj timeAvgMeanU: " << timeAvgMeanU << endl;

    if (checkURef == "yes")
    {
        volVectorField readURef(
            IOobject(
                "URef",
                Foam::name(endTime),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE),
            mesh);

        volVectorField endPtErr("endPtErr", U - readURef);
        Info << "Infinity norm of endPtErr normalized by 1.0:" << this->getMaxAbs(endPtErr) / 1.0 << endl;
        Info << "L2 norm of endPtErr normalized by 1.0:" << this->L2norm(endPtErr) / std::sqrt(U.size()) / 1.0 << endl;
    }

    Info << "End\n"
         << endl;

    return 0;
}

label DAIrkPimpleFoam::runFPAdj(
    Vec dFdW,
    Vec psi)
{
    // If adjoint converged, then adjConv = 0
    // Otherwise, adjConv = 1
    // Set adjConv always 0 for now
    label adjConv = 0;

#ifdef CODI_ADR

    Info << "Solving the adjoint using non-dual fixed-point iteration method..."
         << "  Execution Time: " << meshPtr_->time().elapsedCpuTime() << " s" << endl;

    fvMesh& mesh = meshPtr_();
    const Time& runTime = runTimePtr_;

    // Numerical settings
    word divUScheme = "div(phi,U)";
    word divGradUScheme = "div((nuEff*dev2(T(grad(U)))))";
    word divNuTildaScheme = "div(phi,nuTilda)";

    const fvSolution& myFvSolution = mesh.thisDb().lookupObject<fvSolution>("fvSolution");
    dictionary solverDictU = myFvSolution.subDict("solvers").subDict("U");
    dictionary solverDictP = myFvSolution.subDict("solvers").subDict("p");
    dictionary solverDictNuTilda = myFvSolution.subDict("solvers").subDict("nuTilda");

    // Get endTime, deltaT, and nTimeSteps
    // Note that here we assume uniform deltaT, but deltaT can be non-uniform
    scalar endTime = runTime.endTime().value();
    scalar deltaT = runTime.deltaTValue();
    label nTimeSteps = std::round(endTime / deltaT);

    // Set stepIndex and timeInstance as the last time step
    label stepIndex = nTimeSteps;
    scalar timeInstance = endTime;

#include "createControl.H"
#include "createFieldsIrkPimple.H"
#include "IrkControl.H"

    const objectRegistry& db = meshPtr_->thisDb();

    // Get nu, nut, nuTilda, y
    volScalarField& nu = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nu"));
    volScalarField& nut = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nut"));
    volScalarField& nuTilda = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("nuTilda"));
    volScalarField& y = const_cast<volScalarField&>(mesh.thisDb().lookupObject<volScalarField>("yWall"));

    // Duplicate state variables for stages
    volVectorField U1("U1", U);
    volVectorField U2("U2", U);
    volScalarField p1("p1", p);
    volScalarField p2("p2", p);
    surfaceScalarField phi1("phi1", phi);
    surfaceScalarField phi2("phi2", phi);
    // SA turbulence model
    volScalarField nuTilda1("nuTilda1", nuTilda);
    volScalarField nuTilda2("nuTilda2", nuTilda);
    volScalarField nut1("nut1", nut);
    volScalarField nut2("nut2", nut);

    // Settings for stage pressure
    mesh.setFluxRequired(p1.name());
    mesh.setFluxRequired(p2.name());

    // Initialize primal residuals
#include "initPriRes.H"

    // Initialize all-zero adjoint equation rhs (reversed sign):
    volVectorField mAdjRhsU1("mAdjRhsU1", 0.0 * U);
    volScalarField mAdjRhsP1("mAdjRhsP1", 0.0 * p);
    surfaceScalarField mAdjRhsPhi1("mAdjRhsPhi1", 0.0 * phi);
    volScalarField mAdjRhsNuTilda1("mAdjRhsNuTilda1", 0.0 * nuTilda);

    volVectorField mAdjRhsU2("mAdjRhsU2", mAdjRhsU1);
    volScalarField mAdjRhsP2("mAdjRhsP2", mAdjRhsP1);
    surfaceScalarField mAdjRhsPhi2("mAdjRhsPhi2", mAdjRhsPhi1);
    volScalarField mAdjRhsNuTilda2("mAdjRhsNuTilda2", mAdjRhsNuTilda1);

    // Initialize all-zero adjoint residuals
    volVectorField adjU1Res("adjU1Res", 0.0 * U);
    volScalarField adjP1Res("adjP1Res", 0.0 * p);
    surfaceScalarField adjPhi1Res("adjPhi1Res", 0.0 * phi);
    volScalarField adjNuTilda1Res("adjNuTilda1Res", 0.0 * nuTilda);

    volVectorField adjU2Res("adjU2Res", adjU1Res);
    volScalarField adjP2Res("adjP2Res", adjP1Res);
    surfaceScalarField adjPhi2Res("adjPhi2Res", adjPhi1Res);
    volScalarField adjNuTilda2Res("adjNuTilda2Res", adjNuTilda1Res);

    // Initialize all-zero adjoint vector
    volVectorField U1Psi("U1Psi", 0.0 * U);
    volScalarField p1Psi("p1Psi", 0.0 * p);
    surfaceScalarField phi1Psi("phi1Psi", 0.0 * phi);
    volScalarField nuTilda1Psi("nuTilda1Psi", 0.0 * nuTilda);

    volVectorField U2Psi("U2Psi", U1Psi);
    volScalarField p2Psi("p2Psi", p1Psi);
    surfaceScalarField phi2Psi("phi2Psi", phi1Psi);
    volScalarField nuTilda2Psi("nuTilda2Psi", nuTilda1Psi);

    // Initialize the total gradient dFdX as zero
    scalar dFdX = 0.0;
    label dvPatchI = -100;
    scalar X = 0.0;
    // Note: "inlet" only works for certain cases
    dvPatchI = meshPtr_->boundaryMesh().findPatchID("inlet");
    X = U.boundaryFieldRef()[dvPatchI][0][0];
    Info << "X: " << X << endl;

    // get the reverse-mode AD tape
    codi::RealReverse::Tape& tape = codi::RealReverse::getTape();

    while (stepIndex > 0)
    {
        Info << "Reverse adjoint run for time step = " << stepIndex << endl;

        // Read states for Radau23
#include "readRadau23.H"

        if (checkPriRes == "yes")
        {
            this->calcPriResIrkOrig(U, U1, p1, phi1, nuTilda1, nut1, U2, p2, phi2, nuTilda2, nut2, nu, deltaT, U1Res, p1Res, phi1Res, U2Res, p2Res, phi2Res, relaxUEqn);
            this->calcPriSAResIrkOrig(nuTilda, U1, phi1, nuTilda1, U2, phi2, nuTilda2, y, nu, deltaT, nuTilda1Res, nuTilda2Res);

            Info << "L2 norm of U1Res: " << this->L2norm(U1Res.primitiveField()) << endl;
            Info << "L2 norm of U2Res: " << this->L2norm(U2Res.primitiveField()) << endl;
            Info << "L2 norm of p1Res: " << this->L2norm(p1Res.primitiveField()) << endl;
            Info << "L2 norm of p2Res: " << this->L2norm(p2Res.primitiveField()) << endl;
            Info << "L2 norm of phi1Res: " << this->L2norm(phi1Res, phi1.mesh().magSf()) << endl;
            Info << "L2 norm of phi2Res: " << this->L2norm(phi2Res, phi2.mesh().magSf()) << endl;
            Info << "L2 norm of nuTilda1Res: " << this->L2norm(nuTilda1Res.primitiveField()) << endl;
            Info << "L2 norm of nuTilda2Res: " << this->L2norm(nuTilda2Res.primitiveField()) << endl;
        }

        // Initialize function of interest as zero
        // They are the obj func at the two stages
        scalar funcOfInte1 = 0.0;
        scalar funcOfInte2 = 0.0;

        // The scaling factor below is for the time avg type
        scalar objFuncScale = deltaT / endTime;

        // Get the scaling factors for both stages
        scalar objFuncScaleStage1 = objFuncScale * w1;
        scalar objFuncScaleStage2 = objFuncScale * w2;

        // * * * * * * * * * * * * * * * * * * //
        // Solve with non-dual adjoint

        // Setup the pseudoEqns for both stages
#include "pseudoEqns.H"

        // Calculate dfdw (partial)
        {
#include "adjRhs_dfdw.H"
        }

        // If adjResCnt == 0, adjRes should create the tape, otherwise the tape is reused
        label adjResCnt = 0;

        // Non-dual adjoint iterations with block GS sweeps
        for (label sweepIndex = 0; sweepIndex < maxSweep; sweepIndex++)
        {
            Info << "Block GS sweep: " << sweepIndex + 1 << endl;

            // * * * * * * * * * * * * * * * * * * //
            // 1st stage

            // Calculate adjoint residual
#include "adjRes.H"
            // ------ U1 -----

            // Overwrite the rhs with adjURes, and then solve
            pseudo_U1Eqn.source() = adjU1Res.primitiveField();

            // Make sure that boundary contribution to source is zero,
            // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
            forAll(pseudo_U1.boundaryField(), patchI)
            {
                const fvPatch& pp = pseudo_U1.boundaryField()[patchI].patch();
                forAll(pp, faceI)
                {
                    label cellI = pp.faceCells()[faceI];
                    pseudo_U1Eqn.source()[cellI] -= pseudo_U1Eqn.boundaryCoeffs()[patchI][faceI];
                }
            }

            // Before solve, force xEqn.psi() to be solved into all zero
            forAll(pseudo_U1.primitiveFieldRef(), cellI)
            {
                pseudo_U1.primitiveFieldRef()[cellI][0] = 0;
                pseudo_U1.primitiveFieldRef()[cellI][1] = 0;
                pseudo_U1.primitiveFieldRef()[cellI][2] = 0;
            }

            pseudo_U1Eqn.solve(solverDictU);

            // Apply update
            U1Psi -= pseudo_U1 * relaxU * adjRelaxStage1;

            while (pimple.correct())
            {
                // Calculate adjoint residual
#include "adjRes.H"
                //#include "printAdjResL2.H"
                // ------ p1 -----

                // Overwrite the rhs with adjPRes, and then solve
                pseudo_p1Eqn.source() = adjP1Res.primitiveField();

                pseudo_p1Eqn.setReference(0, 0.0);

                // Make sure that boundary contribution to source is zero,
                // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
                forAll(pseudo_p1.boundaryField(), patchI)
                {
                    const fvPatch& pp = pseudo_p1.boundaryField()[patchI].patch();
                    forAll(pp, faceI)
                    {
                        label cellI = pp.faceCells()[faceI];
                        pseudo_p1Eqn.source()[cellI] -= pseudo_p1Eqn.boundaryCoeffs()[patchI][faceI];
                    }
                }

                // Before solve, force xEqn.psi() to be solved into all zero
                forAll(pseudo_p1.primitiveFieldRef(), cellI)
                {
                    pseudo_p1.primitiveFieldRef()[cellI] = 0;
                }

                pseudo_p1Eqn.solve(solverDictP);

                // Apply update
                p1Psi -= pseudo_p1 * relaxP * adjRelaxStage1;

                // Calculate adjoint residual
#include "adjRes.H"
                //#include "printAdjResL2.H"
                // -------- phi1  --------
                // Apply update
                phi1Psi += adjPhi1Res * relaxPhi * adjRelaxStage1;
            }

            // Calculate adjoint residual
#include "adjRes.H"
            //#include "printAdjResL2.H"
            // ------ nuTilda1 -----

            // Overwrite the rhs with adjNuTildaRes, and then solve
            pseudo_nuTilda1Eqn.source() = adjNuTilda1Res.primitiveField();

            // Make sure that boundary contribution to source is zero,
            // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
            forAll(pseudo_nuTilda1.boundaryField(), patchI)
            {
                const fvPatch& pp = pseudo_nuTilda1.boundaryField()[patchI].patch();
                forAll(pp, faceI)
                {
                    label cellI = pp.faceCells()[faceI];
                    pseudo_nuTilda1Eqn.source()[cellI] -= pseudo_nuTilda1Eqn.boundaryCoeffs()[patchI][faceI];
                }
            }

            // Before solve, force xEqn.psi to be solved into all zero
            forAll(pseudo_nuTilda1.primitiveFieldRef(), cellI)
            {
                pseudo_nuTilda1.primitiveFieldRef()[cellI] = 0;
            }

            pseudo_nuTilda1Eqn.solve(solverDictNuTilda);

            // Apply update
            nuTilda1Psi -= pseudo_nuTilda1 * relaxNuTilda * adjRelaxStage1;

            // * * * * * * * * * * * * * * * * * * //
            // 2nd stage

            // Calculate adjoint residual
#include "adjRes.H"
            //#include "printAdjResL2.H"
            // ------ U2 -----

            // Overwrite the rhs with adjURes, and then solve
            pseudo_U2Eqn.source() = adjU2Res.primitiveField();

            // Make sure that boundary contribution to source is zero,
            // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
            forAll(pseudo_U2.boundaryField(), patchI)
            {
                const fvPatch& pp = pseudo_U2.boundaryField()[patchI].patch();
                forAll(pp, faceI)
                {
                    label cellI = pp.faceCells()[faceI];
                    pseudo_U2Eqn.source()[cellI] -= pseudo_U2Eqn.boundaryCoeffs()[patchI][faceI];
                }
            }

            // Before solve, force xEqn.psi() to be solved into all zero
            forAll(pseudo_U2.primitiveFieldRef(), cellI)
            {
                pseudo_U2.primitiveFieldRef()[cellI][0] = 0;
                pseudo_U2.primitiveFieldRef()[cellI][1] = 0;
                pseudo_U2.primitiveFieldRef()[cellI][2] = 0;
            }

            pseudo_U2Eqn.solve(solverDictU);

            // Apply update
            U2Psi -= pseudo_U2 * relaxU * adjRelaxStage2;

            while (pimple.correct())
            {
                // Calculate adjoint residual
#include "adjRes.H"
                //#include "printAdjResL2.H"
                // ------ p2 -----

                // Overwrite the rhs with adjPRes, and then solve
                pseudo_p2Eqn.source() = adjP2Res.primitiveField();

                pseudo_p2Eqn.setReference(0, 0.0);

                // Make sure that boundary contribution to source is zero,
                // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
                forAll(pseudo_p2.boundaryField(), patchI)
                {
                    const fvPatch& pp = pseudo_p2.boundaryField()[patchI].patch();
                    forAll(pp, faceI)
                    {
                        label cellI = pp.faceCells()[faceI];
                        pseudo_p2Eqn.source()[cellI] -= pseudo_p2Eqn.boundaryCoeffs()[patchI][faceI];
                    }
                }

                // Before solve, force xEqn.psi() to be solved into all zero
                forAll(pseudo_p2.primitiveFieldRef(), cellI)
                {
                    pseudo_p2.primitiveFieldRef()[cellI] = 0;
                }

                pseudo_p2Eqn.solve(solverDictP);

                // Apply update
                p2Psi -= pseudo_p2 * relaxP * adjRelaxStage2;

                // Calculate adjoint residual
#include "adjRes.H"
                //#include "printAdjResL2.H"
                // -------- phi2  --------
                // Apply update
                phi2Psi += adjPhi2Res * relaxPhi * adjRelaxStage2;
            }

            // Calculate adjoint residual
#include "adjRes.H"
            //#include "printAdjResL2.H"
            // ------ nuTilda2 -----

            // Overwrite the rhs with adjNuTildaRes, and then solve
            pseudo_nuTilda2Eqn.source() = adjNuTilda2Res.primitiveField();

            // Make sure that boundary contribution to source is zero,
            // Alternatively, we can deduct source by boundary contribution, so that it would cancel out during solve.
            forAll(pseudo_nuTilda2.boundaryField(), patchI)
            {
                const fvPatch& pp = pseudo_nuTilda2.boundaryField()[patchI].patch();
                forAll(pp, faceI)
                {
                    label cellI = pp.faceCells()[faceI];
                    pseudo_nuTilda2Eqn.source()[cellI] -= pseudo_nuTilda2Eqn.boundaryCoeffs()[patchI][faceI];
                }
            }

            // Before solve, force xEqn.psi to be solved into all zero
            forAll(pseudo_nuTilda2.primitiveFieldRef(), cellI)
            {
                pseudo_nuTilda2.primitiveFieldRef()[cellI] = 0;
            }

            pseudo_nuTilda2Eqn.solve(solverDictNuTilda);

            // Apply update
            nuTilda2Psi -= pseudo_nuTilda2 * relaxNuTilda * adjRelaxStage2;
        }

        // Reset and deactivate input for the adjRes tape
        tape.reset();
#include "deactivateW1.H"
#include "deactivateW2.H"

        Info << "Non-dual adjoint iterations finished,        Execution Time: " << runTime.elapsedCpuTime() << " s" << endl;

        // * * * * * * * * * * * * * * * * * * //
        // After the current step adjoint solve finishes

        // Accumulatively add dfndX to dFdX
        {
#include "dfdx.H"
        }

        // Calculate the rhs contribution to the next adjoint solve
        if (stepIndex > 1)
        {
#include "adjRhs_init.H"
        }

        stepIndex--;
        timeInstance -= deltaT;
    }

    Info << "full dFdX: " << dFdX << endl;

#endif

    return adjConv;
}

} // End namespace Foam

// ************************************************************************* //
