// This code deactive current step state variables (1st stage): U1, p1, phi1, nuTilda1

// Start with U, note that U has 3 components
forAll(U1, cellI)
{
    tape.deactivateValue(U1[cellI][0]);
    tape.deactivateValue(U1[cellI][1]);
    tape.deactivateValue(U1[cellI][2]);
}
// Now deactive p as input:
forAll(p1, cellI)
{
    tape.deactivateValue(p1[cellI]);
}
// Then, deactive phi as input
// Deactive internal of phi:
forAll(phi1.primitiveFieldRef(), faceI)
{
    tape.deactivateValue(phi1.primitiveFieldRef()[faceI]);
}
// Deactive boundary of phi:
forAll(phi1.boundaryFieldRef(), patchI)
{
    forAll(phi1.boundaryFieldRef()[patchI], faceI)
    {
        tape.deactivateValue(phi1.boundaryFieldRef()[patchI][faceI]);
    }
}
// And then, deactive nuTilda as input:
forAll(nuTilda1, cellI)
{
    tape.deactivateValue(nuTilda1[cellI]);
}
