// Calculate the adjoint residual for both stages

// Pass mAdjRhs to adjRes:
adjU1Res = mAdjRhsU1;
adjP1Res = mAdjRhsP1;
adjPhi1Res = mAdjRhsPhi1;
adjNuTilda1Res = mAdjRhsNuTilda1;

adjU2Res = mAdjRhsU2;
adjP2Res = mAdjRhsP2;
adjPhi2Res = mAdjRhsPhi2;
adjNuTilda2Res = mAdjRhsNuTilda2;

// Build the tape if adjResCnt == 0
if (adjResCnt == 0)
{
    tape.setActive();

    // register U1, p1, phi1 as input
    // Start with U, note that U has 3 components
    forAll(U1, cellI)
    {
        tape.registerInput(U1[cellI][0]);
        tape.registerInput(U1[cellI][1]);
        tape.registerInput(U1[cellI][2]);
    }
    // Now register p as input:
    forAll(p1, cellI)
    {
        tape.registerInput(p1[cellI]);
    }
    // Then, register phi as input
    // Note that both the internal field of phi and the boundary fields need to be registered
    // Register internal of phi:
    forAll(phi1.primitiveFieldRef(), faceI)
    {
        tape.registerInput(phi1.primitiveFieldRef()[faceI]);
    }
    // Register boundary of phi:
    forAll(phi1.boundaryFieldRef(), patchI)
    {
        forAll(phi1.boundaryFieldRef()[patchI], faceI)
        {
            tape.registerInput(phi1.boundaryFieldRef()[patchI][faceI]);
        }
    }
    // And then, register turbulence variable nuTilda as input:
    forAll(nuTilda1, cellI)
    {
        tape.registerInput(nuTilda1[cellI]);
    }

    // register U2, p2, phi2 as input
    // Start with U, note that U has 3 components
    forAll(U2, cellI)
    {
        tape.registerInput(U2[cellI][0]);
        tape.registerInput(U2[cellI][1]);
        tape.registerInput(U2[cellI][2]);
    }
    // Now register p as input:
    forAll(p2, cellI)
    {
        tape.registerInput(p2[cellI]);
    }
    // Then, register phi as input
    // Note that both the internal field of phi and the boundary fields need to be registered
    // Register internal of phi:
    forAll(phi2.primitiveFieldRef(), faceI)
    {
        tape.registerInput(phi2.primitiveFieldRef()[faceI]);
    }
    // Register boundary of phi:
    forAll(phi2.boundaryFieldRef(), patchI)
    {
        forAll(phi2.boundaryFieldRef()[patchI], faceI)
        {
            tape.registerInput(phi2.boundaryFieldRef()[patchI][faceI]);
        }
    }
    // And then, register turbulence variable nuTilda as input:
    forAll(nuTilda2, cellI)
    {
        tape.registerInput(nuTilda2[cellI]);
    }

    // Link intermediate steps
    forAll(U.boundaryFieldRef()[dvPatchI], cellI)
    {
        U.boundaryFieldRef()[dvPatchI][cellI][0] = X;
    }
    forAll(U1.boundaryFieldRef()[dvPatchI], cellI)
    {
        U1.boundaryFieldRef()[dvPatchI][cellI][0] = X;
    }
    forAll(U2.boundaryFieldRef()[dvPatchI], cellI)
    {
        U2.boundaryFieldRef()[dvPatchI][cellI][0] = X;
    }

    /*
    U.correctBoundaryConditions();
    U1.correctBoundaryConditions();
    p1.correctBoundaryConditions();
    U2.correctBoundaryConditions();
    p2.correctBoundaryConditions();
    */

    // Calculate the primal residuals scaled with mesh volume
    this->scaledPriResIrk(U, nuTilda, U1, p1, phi1, nuTilda1, nut1, U2, p2, phi2, nuTilda2, nut2, y, nu, deltaT, U1Res, p1Res, phi1Res, nuTilda1Res, U2Res, p2Res, phi2Res, nuTilda2Res, relaxUEqn);

    // Register the output U1Res, p1Res, phi1Res:
    // Start with URes, note that URes has 3 components
    forAll(U1Res, cellI)
    {
        tape.registerOutput(U1Res[cellI][0]);
        tape.registerOutput(U1Res[cellI][1]);
        tape.registerOutput(U1Res[cellI][2]);
    }
    // Now register pRes as output:
    forAll(p1Res, cellI)
    {
        tape.registerOutput(p1Res[cellI]);
    }
    // Then, register phiRes as output
    // Note that both the internal field of phiRes and the boundary fields need to be registered
    // Register internal of phiRes:
    forAll(phi1Res.primitiveFieldRef(), faceI)
    {
        tape.registerOutput(phi1Res.primitiveFieldRef()[faceI]);
    }
    // Register boundary of phiRes:
    forAll(phi1Res.boundaryFieldRef(), patchI)
    {
        forAll(phi1Res.boundaryFieldRef()[patchI], faceI)
        {
            tape.registerOutput(phi1Res.boundaryFieldRef()[patchI][faceI]);
        }
    }
    // Register the output nuTildaRes:
    forAll(nuTilda1Res, cellI)
    {
        tape.registerOutput(nuTilda1Res[cellI]);
    }

    // Register the output U2Res, p2Res, phi2Res:
    // Start with URes, note that URes has 3 components
    forAll(U2Res, cellI)
    {
        tape.registerOutput(U2Res[cellI][0]);
        tape.registerOutput(U2Res[cellI][1]);
        tape.registerOutput(U2Res[cellI][2]);
    }
    // Now register pRes as output:
    forAll(p2Res, cellI)
    {
        tape.registerOutput(p2Res[cellI]);
    }
    // Then, register phiRes as output
    // Note that both the internal field of phiRes and the boundary fields need to be registered
    // Register internal of phiRes:
    forAll(phi2Res.primitiveFieldRef(), faceI)
    {
        tape.registerOutput(phi2Res.primitiveFieldRef()[faceI]);
    }
    // Register boundary of phiRes:
    forAll(phi2Res.boundaryFieldRef(), patchI)
    {
        forAll(phi2Res.boundaryFieldRef()[patchI], faceI)
        {
            tape.registerOutput(phi2Res.boundaryFieldRef()[patchI][faceI]);
        }
    }
    // Register the output nuTildaRes:
    forAll(nuTilda2Res, cellI)
    {
        tape.registerOutput(nuTilda2Res[cellI]);
    }

    // stop recording
    tape.setPassive();
}

// set the AD seed to the output var, 1st stage
// Start with URes, note that URes has 3 components
forAll(U1Res, cellI)
{
    U1Res[cellI][0].setGradient(U1Psi[cellI][0].getValue());
    U1Res[cellI][1].setGradient(U1Psi[cellI][1].getValue());
    U1Res[cellI][2].setGradient(U1Psi[cellI][2].getValue());
}
// Now seed pRes:
forAll(p1Res, cellI)
{
    p1Res[cellI].setGradient(p1Psi[cellI].getValue());
}
// Then, seed phiRes:
// Seed internal of phiRes:
forAll(phi1Res.primitiveFieldRef(), faceI)
{
    phi1Res.primitiveFieldRef()[faceI].setGradient(phi1Psi.primitiveFieldRef()[faceI].getValue());
}
// Seed boundary of phiRes:
forAll(phi1Res.boundaryFieldRef(), patchI)
{
    forAll(phi1Res.boundaryFieldRef()[patchI], faceI)
    {
        phi1Res.boundaryFieldRef()[patchI][faceI].setGradient(phi1Psi.boundaryFieldRef()[patchI][faceI].getValue());
    }
}
// And then, seed nuTildaRes:
forAll(nuTilda1Res, cellI)
{
    nuTilda1Res[cellI].setGradient(nuTilda1Psi[cellI].getValue());
}

// set the AD seed to the output var, 2nd stage
// Start with URes, note that URes has 3 components
forAll(U2Res, cellI)
{
    U2Res[cellI][0].setGradient(U2Psi[cellI][0].getValue());
    U2Res[cellI][1].setGradient(U2Psi[cellI][1].getValue());
    U2Res[cellI][2].setGradient(U2Psi[cellI][2].getValue());
}
// Now seed pRes:
forAll(p2Res, cellI)
{
    p2Res[cellI].setGradient(p2Psi[cellI].getValue());
}
// Then, seed phiRes:
// Seed internal of phiRes:
forAll(phi2Res.primitiveFieldRef(), faceI)
{
    phi2Res.primitiveFieldRef()[faceI].setGradient(phi2Psi.primitiveFieldRef()[faceI].getValue());
}
// Seed boundary of phiRes:
forAll(phi2Res.boundaryFieldRef(), patchI)
{
    forAll(phi2Res.boundaryFieldRef()[patchI], faceI)
    {
        phi2Res.boundaryFieldRef()[patchI][faceI].setGradient(phi2Psi.boundaryFieldRef()[patchI][faceI].getValue());
    }
}
// And then, seed nuTildaRes:
forAll(nuTilda2Res, cellI)
{
    nuTilda2Res[cellI].setGradient(nuTilda2Psi[cellI].getValue());
}

// evaluate the tape to compute the derivative of the seeded output wrt all the input
tape.evaluate();

forAll(U1, cellI)
{
    adjU1Res[cellI][0] += U1[cellI][0].getGradient();
    adjU1Res[cellI][1] += U1[cellI][1].getGradient();
    adjU1Res[cellI][2] += U1[cellI][2].getGradient();
}
forAll(p1, cellI)
{
    adjP1Res[cellI] += p1[cellI].getGradient();
}
forAll(phi1.primitiveFieldRef(), faceI)
{
    adjPhi1Res.primitiveFieldRef()[faceI] += phi1.primitiveFieldRef()[faceI].getGradient();
}
forAll(phi1.boundaryFieldRef(), patchI)
{
    forAll(phi1.boundaryFieldRef()[patchI], faceI)
    {
        adjPhi1Res.boundaryFieldRef()[patchI][faceI] += phi1.boundaryFieldRef()[patchI][faceI].getGradient();
    }
}
forAll(nuTilda1, cellI)
{
    adjNuTilda1Res[cellI] += nuTilda1[cellI].getGradient();
}

forAll(U2, cellI)
{
    adjU2Res[cellI][0] += U2[cellI][0].getGradient();
    adjU2Res[cellI][1] += U2[cellI][1].getGradient();
    adjU2Res[cellI][2] += U2[cellI][2].getGradient();
}
forAll(p2, cellI)
{
    adjP2Res[cellI] += p2[cellI].getGradient();
}
forAll(phi2.primitiveFieldRef(), faceI)
{
    adjPhi2Res.primitiveFieldRef()[faceI] += phi2.primitiveFieldRef()[faceI].getGradient();
}
forAll(phi2.boundaryFieldRef(), patchI)
{
    forAll(phi2.boundaryFieldRef()[patchI], faceI)
    {
        adjPhi2Res.boundaryFieldRef()[patchI][faceI] += phi2.boundaryFieldRef()[patchI][faceI].getGradient();
    }
}
forAll(nuTilda2, cellI)
{
    adjNuTilda2Res[cellI] += nuTilda2[cellI].getGradient();
}

// Clear adjoints for future Jacobian calculations
tape.clearAdjoints();

adjResCnt++;
