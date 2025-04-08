// Additional term if not the last time
// dR^n/dw^(n-1) * psi^n
// Note that for Radau schemes, the oldTime variable is U (and nuTilda if turb SA)
// R^n has no dependency on other oldTime variables
// U read from the last time step is its 2nd stage.

tape.setActive();

// Start with U, note that U has 3 components
forAll(U, cellI)
{
    tape.registerInput(U[cellI][0]);
    tape.registerInput(U[cellI][1]);
    tape.registerInput(U[cellI][2]);
}
// And then, register turbulence variable nuTilda as input:
forAll(nuTilda, cellI)
{
    tape.registerInput(nuTilda[cellI]);
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
// get dRdU0, note that it has 3 components
forAll(U, cellI)
{
    mAdjRhsU2[cellI][0] = U[cellI][0].getGradient();
    mAdjRhsU2[cellI][1] = U[cellI][1].getGradient();
    mAdjRhsU2[cellI][2] = U[cellI][2].getGradient();
}
// get dRdNuTilda0
forAll(nuTilda, cellI)
{
    mAdjRhsNuTilda2[cellI] = nuTilda[cellI].getGradient();
}

// Clear adjoints and reset the tape for future recording.
tape.clearAdjoints();
tape.reset();

// Deactivate input variables after usage
// Deactivate oldTime U
forAll(U, cellI)
{
    tape.deactivateValue(U[cellI][0]);
    tape.deactivateValue(U[cellI][1]);
    tape.deactivateValue(U[cellI][2]);
}
// And then, deactive oldTime nuTilda as input:
forAll(nuTilda, cellI)
{
    tape.deactivateValue(nuTilda[cellI]);
}

// Set all other components of mAdjRhs to zero
mAdjRhsU1 = 0.0 * U;
mAdjRhsP1 = 0.0 * p;
mAdjRhsPhi1 = 0.0 * phi;
mAdjRhsNuTilda1 =  0.0 * nuTilda;
mAdjRhsP2 = 0.0 * p;
mAdjRhsPhi2 = 0.0 * phi;
