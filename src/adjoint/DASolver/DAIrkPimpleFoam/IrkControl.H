// get IRKDict settings, default to Radau23 for now
IOdictionary IRKDict(
    IOobject(
        "IRKDict",
        mesh.time().system(),
        mesh,
        IOobject::READ_IF_PRESENT,
        IOobject::NO_WRITE));

scalar relaxU = 1.0;
if (IRKDict.found("relaxU"))
{
    if (IRKDict.getScalar("relaxU") > 0)
    {
        relaxU = IRKDict.getScalar("relaxU");
    }
}

scalar relaxP = 1.0;
if (IRKDict.found("relaxP"))
{
    if (IRKDict.getScalar("relaxP") > 0)
    {
        relaxP = IRKDict.getScalar("relaxP");
    }
}

scalar relaxPhi = 1.0;
if (IRKDict.found("relaxPhi"))
{
    if (IRKDict.getScalar("relaxPhi") > 0)
    {
        relaxPhi = IRKDict.getScalar("relaxPhi");
    }
}

scalar relaxNuTilda = 1.0;
if (IRKDict.found("relaxNuTilda"))
{
    if (IRKDict.getScalar("relaxNuTilda") > 0)
    {
        relaxNuTilda = IRKDict.getScalar("relaxNuTilda");
    }
}

scalar relaxStage1 = 0.8;
if (IRKDict.found("relaxStage1"))
{
    if (IRKDict.getScalar("relaxStage1") > 0)
    {
        relaxStage1 = IRKDict.getScalar("relaxStage1");
    }
}

scalar relaxStage2 = 0.8;
if (IRKDict.found("relaxStage2"))
{
    if (IRKDict.getScalar("relaxStage2") > 0)
    {
        relaxStage2 = IRKDict.getScalar("relaxStage2");
    }
}

scalar relaxUEqn = 1.0;
scalar relaxNuTildaEqn = 1.0;
if (IRKDict.found("relaxNuTildaEqn"))
{
    if (IRKDict.getScalar("relaxNuTildaEqn") > 0)
    {
        relaxNuTildaEqn = IRKDict.getScalar("relaxNuTildaEqn");
    }
}

label maxSweep = 10;
if (IRKDict.found("maxSweep"))
{
    if (IRKDict.getLabel("maxSweep") > 0)
    {
        maxSweep = IRKDict.getLabel("maxSweep");
    }
}

word checkPriRes = "yes";
if (IRKDict.found("checkPriRes"))
{
    checkPriRes = IRKDict.getWord("checkPriRes");

    Info << "If checking primal residual: " << checkPriRes << endl;

    if (checkPriRes != "yes" && checkPriRes != "no")
    {
        Info << "checkPriRes not recognized, Using the default: yes" << endl
             << "Options are: yes, no" << endl;
        checkPriRes = "yes";
    }
}
