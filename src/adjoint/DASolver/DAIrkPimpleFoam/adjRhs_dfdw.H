// Adding dfdw (partial) to the rhs of the adj eqn
// We can deal one stage at a time

// * * * * * * * * * * * * * * * * * * //
// 1st stage

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

// Link intermediate steps
forAll(U1.boundaryFieldRef()[dvPatchI], cellI)
{
    U1.boundaryFieldRef()[dvPatchI][cellI][0] = X;
}
U1.correctBoundaryConditions();
p1.correctBoundaryConditions();
nuTilda1.correctBoundaryConditions();

// Calculate the obj func
funcOfInte1 = calcMeanU(U1);

// register the output
tape.registerOutput(funcOfInte1);
// stop recording
tape.setPassive();
// set the AD seed to the output var
// Also apply the scaling factor for the objective function type
funcOfInte1.setGradient(objFuncScaleStage1.getValue());

// evaluate the tape to compute the derivative of the seeded output wrt all the input
tape.evaluate();
// get dFdU, note that it has 3 components
forAll(U1, cellI)
{
    mAdjRhsU1[cellI][0] -= U1[cellI][0].getGradient();
    mAdjRhsU1[cellI][1] -= U1[cellI][1].getGradient();
    mAdjRhsU1[cellI][2] -= U1[cellI][2].getGradient();
}
// get dFdP
forAll(p1, cellI)
{
    mAdjRhsP1[cellI] -= p1[cellI].getGradient();
}
// get dFdPhi
forAll(phi1.primitiveFieldRef(), faceI)
{
    mAdjRhsPhi1.primitiveFieldRef()[faceI] -= phi1.primitiveFieldRef()[faceI].getGradient();
}
forAll(phi1.boundaryFieldRef(), patchI)
{
    forAll(phi1.boundaryFieldRef()[patchI], faceI)
    {
        mAdjRhsPhi1.boundaryFieldRef()[patchI][faceI] -= phi1.boundaryFieldRef()[patchI][faceI].getGradient();
    }
}
// get dFdNuTilda
forAll(nuTilda1, cellI)
{
    mAdjRhsNuTilda1[cellI] -= nuTilda1[cellI].getGradient();
}

// Clear adjoints and reset the tape for future recording.
tape.clearAdjoints();
tape.reset();

// Deactivate input variables after usage
// Deactivate U, p, phi as ADR tape inputs
#include "deactivateW1.H"

// * * * * * * * * * * * * * * * * * * //
// 2nd stage

tape.setActive();

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
forAll(U2.boundaryFieldRef()[dvPatchI], cellI)
{
    U2.boundaryFieldRef()[dvPatchI][cellI][0] = X;
}
U2.correctBoundaryConditions();
p2.correctBoundaryConditions();
nuTilda2.correctBoundaryConditions();

// Calculate the obj func
funcOfInte2 = calcMeanU(U2);

// register the output
tape.registerOutput(funcOfInte2);
// stop recording
tape.setPassive();
// set the AD seed to the output var
// Also apply the scaling factor for the objective function type
funcOfInte2.setGradient(objFuncScaleStage2.getValue());

// evaluate the tape to compute the derivative of the seeded output wrt all the input
tape.evaluate();
// get dFdU, note that it has 3 components
forAll(U2, cellI)
{
    mAdjRhsU2[cellI][0] -= U2[cellI][0].getGradient();
    mAdjRhsU2[cellI][1] -= U2[cellI][1].getGradient();
    mAdjRhsU2[cellI][2] -= U2[cellI][2].getGradient();
}
// get dFdP
forAll(p2, cellI)
{
    mAdjRhsP2[cellI] -= p2[cellI].getGradient();
}
// get dFdPhi
forAll(phi2.primitiveFieldRef(), faceI)
{
    mAdjRhsPhi2.primitiveFieldRef()[faceI] -= phi2.primitiveFieldRef()[faceI].getGradient();
}
forAll(phi2.boundaryFieldRef(), patchI)
{
    forAll(phi2.boundaryFieldRef()[patchI], faceI)
    {
        mAdjRhsPhi2.boundaryFieldRef()[patchI][faceI] -= phi2.boundaryFieldRef()[patchI][faceI].getGradient();
    }
}
// get dFdNuTilda
forAll(nuTilda2, cellI)
{
    mAdjRhsNuTilda2[cellI] -= nuTilda2[cellI].getGradient();
}

// Clear adjoints and reset the tape for future recording.
tape.clearAdjoints();
tape.reset();

// Deactivate input variables after usage
// Deactivate U, p, phi as ADR tape inputs
#include "deactivateW2.H"
