R=0.004; B=8; E0=1.; N=128;
Group {
  Air=Region[1]; Inner=Region[2]; Outer=Region[3];
  Whole=Region[{1,2,3}];
}
Constraint {
  { Name Potential; Type Assign;
    Case { { Region Inner; Value E0*Z[]; } }
  }
}
Jacobian {
  { Name JV; Case { { Region All; Jacobian Vol; } } }
  { Name JS; Case { { Region All; Jacobian Sur; } } }
}
Integration {
  { Name Int;
    Case { { Type Gauss;
      Case { { GeoElement Tetrahedron; NumberOfPoints 15; }
             { GeoElement Triangle; NumberOfPoints 7; }
             { GeoElement Tetrahedron2; NumberOfPoints 15; }
             { GeoElement Triangle2; NumberOfPoints 7; } }
    } }
  }
}
FunctionSpace {
  { Name V; Type Form0;
    BasisFunction {
      { Name sn; NameOfCoef vn; Function BF_Node;
        Support Whole; Entity NodesOf[All]; }
    }
    Constraint {
      { NameOfCoef vn; EntityType NodesOf; NameOfConstraint Potential; }
    }
  }
}
Formulation {
  { Name Laplace; Type FemEquation;
    Quantity { { Name v; Type Local; NameOfSpace V; } }
    Equation {
      Integral { [ Dof{d v}, {d v} ]; In Air; Jacobian JV; Integration Int; }
      Integral { [ 2/(B*R)*Dof{v}, {v} ]; In Outer; Jacobian JS; Integration Int; }
    }
  }
}
Resolution {
  { Name SolveSphere;
    System { { Name A; NameOfFormulation Laplace; } }
    Operation { Generate[A]; Solve[A]; SaveSolution[A]; }
  }
}
PostProcessing {
  { Name Fields; NameOfFormulation Laplace;
    Quantity {
      { Name E; Value { Local { [ Vector[0,0,E0]-{d v} ]; In Air; Jacobian JV; } } }
      { Name Phi; Value { Local { [ {v}-E0*Z[] ]; In Air; Jacobian JV; } } }
    }
  }
}
PostOperation {
  { Name Sample; NameOfPostProcessing Fields;
    Operation {
      Print[E, OnGrid {1.1*R*Sqrt[1-(1-2*($A+0.5)/N)^2]*Cos[Pi*(3-Sqrt[5])*$A],
                       1.1*R*Sqrt[1-(1-2*($A+0.5)/N)^2]*Sin[Pi*(3-Sqrt[5])*$A],
                       1.1*R*(1-2*($A+0.5)/N)} {0:N-1,0,0}, Format Table, File "E_110.txt"];
      Print[Phi, OnGrid {1.1*R*Sqrt[1-(1-2*($A+0.5)/N)^2]*Cos[Pi*(3-Sqrt[5])*$A],
                       1.1*R*Sqrt[1-(1-2*($A+0.5)/N)^2]*Sin[Pi*(3-Sqrt[5])*$A],
                       1.1*R*(1-2*($A+0.5)/N)} {0:N-1,0,0}, Format Table, File "Phi_110.txt"];
    }
  }
}
