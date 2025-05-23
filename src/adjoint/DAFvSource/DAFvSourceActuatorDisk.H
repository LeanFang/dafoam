/*---------------------------------------------------------------------------*\

    DAFoam  : Discrete Adjoint with OpenFOAM
    Version : v4

    Description:
        Child class for ActuatorDisk disk source

\*---------------------------------------------------------------------------*/

#ifndef DAFvSourceActuatorDisk_H
#define DAFvSourceActuatorDisk_H

#include "DAFvSource.H"
#include "addToRunTimeSelectionTable.H"
#include "DAGlobalVar.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
      Class DAFvSourceActuatorDisk Declaration
\*---------------------------------------------------------------------------*/

class DAFvSourceActuatorDisk
    : public DAFvSource
{

protected:
    /// HashTable that contains lists of cell indices that are within the actuator disk space
    HashTable<labelList> fvSourceCellIndices_;

    /// calculate DAFvSourceActuatorDisk::fvSourceCellIndices_
    void calcFvSourceCellIndices(HashTable<labelList>& fvSourceCellIndices);

    /// print interval for primal and adjoint
    label printInterval_;

public:
    TypeName("actuatorDisk");
    // Constructors

    //- Construct from components
    DAFvSourceActuatorDisk(
        const word modelType,
        const fvMesh& mesh,
        const DAOption& daOption,
        const DAModel& daModel,
        const DAIndex& daIndex);

    //- Destructor
    virtual ~DAFvSourceActuatorDisk()
    {
    }

    /// compute the FvSource term
    virtual void calcFvSource(volVectorField& fvSource);

    /// Initialize the values for all types of fvSource in DAGlobalVar, including actuatorDiskPars, heatSourcePars, etc
    virtual void initFvSourcePars();
};

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
