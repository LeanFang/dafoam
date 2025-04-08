// This code first calculates the partial dfdx (1st and 2nd stages)
// Then it calculates the full dfdx by chain rule with the adjoint vector (1st and 2nd stages)

// --- Calculate the partial dfdx, 1st and 2nd stages
tape.setActive();

// Register input X
tape.registerInput(X);

// Link intermediate steps, 1st stage
forAll(U1.boundaryFieldRef()[dvPatchI], cellI)
{
    U1.boundaryFieldRef()[dvPatchI][cellI][0] = X;
}
U1.correctBoundaryConditions();
p1.correctBoundaryConditions();
nuTilda1.correctBoundaryConditions();

// Link intermediate steps, 2nd stage
forAll(U2.boundaryFieldRef()[dvPatchI], cellI)
{
    U2.boundaryFieldRef()[dvPatchI][cellI][0] = X;
}
U2.correctBoundaryConditions();
p2.correctBoundaryConditions();
nuTilda2.correctBoundaryConditions();

// Calculate the obj func, 1st and 2nd stages
funcOfInte1 = calcMeanU(U1);
funcOfInte2 = calcMeanU(U2);

// register the output, 1st and 2nd stages
tape.registerOutput(funcOfInte1);
tape.registerOutput(funcOfInte2);
// stop recording
tape.setPassive();
// set the AD seed to the output var
// Also apply the scaling factor for the objective function type
funcOfInte1.setGradient(objFuncScaleStage1.getValue());
funcOfInte2.setGradient(objFuncScaleStage2.getValue());

// evaluate the tape to compute the derivative of the seeded output wrt all the input
tape.evaluate();
dFdX += X.getGradient();

// reset the tape for future recording
tape.clearAdjoints();
tape.reset();

// Deactivate input variables after usage
tape.deactivateValue(X);

// --- Calculate the full dfdx, 1st and 2nd stages
tape.setActive();

// Register input X
tape.registerInput(X);

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

// Evaluate the tape to compute the derivative of the seeded output wrt all the input
tape.evaluate();
dFdX -= X.getGradient();

// reset the tape for future recording
tape.clearAdjoints();
tape.reset();

// Deactivate input variables after usage
tape.deactivateValue(X);

//Info << "full dFdX: " << dFdX << endl;
