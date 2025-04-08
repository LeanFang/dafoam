// This code deactive current step state variables (2nd stage): U2, p2, phi2, nuTilda2

// Start with U, note that U has 3 components
forAll(U2, cellI)
{
    tape.deactivateValue(U2[cellI][0]);
    tape.deactivateValue(U2[cellI][1]);
    tape.deactivateValue(U2[cellI][2]);
}
// Now deactive p as input:
forAll(p2, cellI)
{
    tape.deactivateValue(p2[cellI]);
}
// Then, deactive phi as input
// Deactive internal of phi:
forAll(phi2.primitiveFieldRef(), faceI)
{
    tape.deactivateValue(phi2.primitiveFieldRef()[faceI]);
}
// Deactive boundary of phi:
forAll(phi2.boundaryFieldRef(), patchI)
{
    forAll(phi2.boundaryFieldRef()[patchI], faceI)
    {
        tape.deactivateValue(phi2.boundaryFieldRef()[patchI][faceI]);
    }
}
// And then, deactive nuTilda as input:
forAll(nuTilda2, cellI)
{
    tape.deactivateValue(nuTilda2[cellI]);
}
