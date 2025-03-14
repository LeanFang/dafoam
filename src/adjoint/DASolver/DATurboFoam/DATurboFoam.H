/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for DATurboFoam
        A steady-state compressible solver for turbomachinery
        This class is modified based on TCFD redSolver (www.cfdsupport.com)

\*---------------------------------------------------------------------------*/

#ifndef DATurboFoam_H
#define DATurboFoam_H

#include "DASolver.H"
#include "addToRunTimeSelectionTable.H"
#include "fluidThermo.H"
#include "turbulentFluidThermoModel.H"
#include "simpleControl.H"
#include "pressureControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DATurboFoam Declaration
\*---------------------------------------------------------------------------*/

class DATurboFoam
    : public DASolver
{

protected:
    /// simple pointer
    autoPtr<simpleControl> simplePtr_;

    /// thermal model
    autoPtr<fluidThermo> pThermoPtr_;

    /// pressure field pointer
    autoPtr<volScalarField> pPtr_;

    /// density field pointer
    autoPtr<volScalarField> rhoPtr_;

    /// velocity field pointer
    autoPtr<volVectorField> UPtr_;

    /// relative velocity field pointer
    autoPtr<volVectorField> URelPtr_;

    /// surface flux field pointer
    autoPtr<surfaceScalarField> phiPtr_;

    /// pressure control pointer
    autoPtr<pressureControl> pressureControlPtr_;

    /// turbulence pointer
    autoPtr<compressible::turbulenceModel> turbulencePtr_;

    /// DATurbulenceModel pointer
    autoPtr<DATurbulenceModel> daTurbulenceModelPtr_;

    /// MRF pointer
    autoPtr<IOMRFZoneListDF> MRFPtr_;

    /// initial mass
    dimensionedScalar initialMass_;

    /// continuity error
    scalar cumulativeContErr_ = 0.0;

public:
    TypeName("DATurboFoam");
    // Constructors

    //- Construct from components
    DATurboFoam(
        char* argsAll,
        PyObject* pyOptions);

    //- Destructor
    virtual ~DATurboFoam()
    {
    }

    /// initialize fields and variables
    virtual void initSolver();

    /// solve the primal equations
    virtual label solvePrimal();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
