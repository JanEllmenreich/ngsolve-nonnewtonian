#!/usr/bin/env python3
from netgen.geom2d import *
from netgen.meshing import *
from netgen.csg import Pnt
import ngsolve
from netgen.meshing import *
from NonNewtonianModels import Bingham
import numpy as np

def CreateSquareMesh(options,nn_model):
    #Parameters
    L = float(options["L"])
    H = float(options["H"])    
    hmax = float(options["hmax"])

    #Geometry
    geo = SplineGeometry()
    pnts = [ (0,-H/2), (L,-H/2), (L,H/2), (0,H/2) ]
    pnums = [geo.AppendPoint(*p) for p in pnts]

    #Connect vertices
    geo.Append ( ["line", pnums[0], pnums[1]],bc="wall",leftdomain=1,rightdomain=0)
    geo.Append ( ["line", pnums[2], pnums[3]], bc="boundary",leftdomain=1,rightdomain=0)
    if options["benchmark"] == "periodic":
        lright = geo.Append ( ["line", pnums[1], pnums[2]], bc="periodic_xm",leftdomain=1,rightdomain=0)
        geo.Append ( ["line", pnums[0], pnums[3]], leftdomain=0, rightdomain=1, copy=lright, bc="periodic_xs")
    else:
        geo.Append ( ["line", pnums[1], pnums[2]], bc="wall",leftdomain=1,rightdomain=0)
        geo.Append ( ["line", pnums[3], pnums[0]], leftdomain=1, rightdomain=0, bc="wall")
    
    if options["structured"]:
        mesh = Mesh()
        mesh.SetGeometry(geo)
        mesh.dim=2
        N_x = int(L/hmax); N_y = int(H/hmax)
        x_arr = np.linspace(0,L,N_x,dtype=np.float64)
        y_arr = np.linspace(-H/2,H/2,N_y,dtype=np.float64)
        pids = []       
        slave = []
        master = []

        for i in range(N_y):
            for j in range(N_x):
                pids.append(mesh.Add (MeshPoint(Pnt(float(x_arr[j]),float(y_arr[i]),0))))
                if options["benchmark"] == "periodic":                       
                    if j == 0:
                        slave.append(pids[-1])
                    if j == N_x-1:
                        master.append(pids[-1])        
        if options["benchmark"] == "periodic":            
            for j in range(len(slave)):        
                mesh.AddPointIdentification(master[j],slave[j],identnr=1,type=2)                                       

        # mesh.Add(FaceDescriptor(surfnr=1,domin=1,bc=1))
        idx_dom = mesh.AddRegion("dom", dim=2)
        idx_top = mesh.AddRegion("boundary", dim=1)
        idx_bot = mesh.AddRegion("wall", dim=1)
        if options["benchmark"] == "periodic":
            idx_left = mesh.AddRegion("periodic_xs", dim=1)
            idx_right  = mesh.AddRegion("periodic_xm", dim=1)
        else:
            idx_left = mesh.AddRegion("wall", dim=1)
            idx_right  = mesh.AddRegion("wall", dim=1)
        
        for i in range(N_y-1):
            for j in range(N_x-1):
                base = i * N_x + j
                pnum1 = [base,base+1,base+N_x]
                pnum2 = [base+1,base+N_x+1,base+N_x]
                elpids1 = [pids[p] for p in pnum1]
                elpids2 = [pids[p] for p in pnum2]
                mesh.Add(Element2D(idx_dom,elpids1)) 
                mesh.Add(Element2D(idx_dom,elpids2))                          

        for i in range(N_x-1):
            mesh.Add(Element1D([pids[i], pids[i+1]], index=idx_bot))
            mesh.Add(Element1D([pids[(N_y-1)*N_x+i], pids[(N_y-1)*N_x+i+1]], index=idx_top))
        for i in range(N_y-1):
            mesh.Add(Element1D([pids[i*N_x+N_x-1], pids[(i+1)*N_x+N_x-1]], index=idx_right))
            mesh.Add(Element1D([pids[(i)*N_x], pids[(i+1)*N_x]], index=idx_left))
    else:
        mesh = geo.GenerateMesh(maxh=hmax)

    return ngsolve.Mesh(mesh)

def CreateCylinderMesh(options,nnmodel):
    H = 0.41; L = 2.0
    mid = (0.2,0.2); r = 0.05

    geo = SplineGeometry()
    geo.AddRectangle( (0, 0), (L, H), bcs = ("wall", "outlet", "wall", "boundary"))
    geo.AddCircle ( mid, r=r, leftdomain=0, rightdomain=1, bc="wall", maxh=0.02)
    mesh = ngsolve.Mesh(geo.GenerateMesh(maxh=options["hmax"]))
    mesh.Curve(int(options["order"]))
    return mesh

def Barycentric_Refinement(template_mesh,options):
    boundaries = template_mesh.GetBoundaries()
    template_mesh = template_mesh.ngmesh
    newmesh = Mesh()
    newmesh.SetGeometry(template_mesh.GetGeometry())
    newmesh.dim = 2

    #Boundary names
    idx_dom = newmesh.AddRegion("dom", dim=2)
    regions = {}
    for i in range(len(boundaries)):
        regions[i] = newmesh.AddRegion(boundaries[i], dim=1)

    #Set MeshPoints
    pids = []
    for p in template_mesh.Points():
        pids.append(newmesh.Add(MeshPoint(Pnt(p.p[0],p.p[1],p.p[2]))))

    #Calculate MidPoint
    def average_pnt(elem_id):
        coords = np.empty((0,3),dtype=np.float64)
        for e in elem_id:
            coords = np.append(coords,[[i for i in newmesh.Points()[e].p]],axis=0)
        average = np.sum(coords,axis=0)/coords.shape[0]
        return Pnt(average[0], average[1], average[2])

    #Append 2D Elements
    for el in template_mesh.Elements2D():
        v1,v2,v3 = el.vertices
        v123 = newmesh.Add(MeshPoint(average_pnt(el.vertices)))
        newmesh.Add(Element2D(idx_dom,[v1,v2,v123]))
        newmesh.Add(Element2D(idx_dom,[v2,v3,v123]))
        newmesh.Add(Element2D(idx_dom,[v3,v1,v123]))

    #Append 1D Elements and define periodic boundary if needed
    master = []
    slave = []
    idx_xm = [i+1 for i, e in enumerate(boundaries) if e == "periodic_xm"]
    idx_xs = [i+1 for i, e in enumerate(boundaries) if e == "periodic_xs"]
    for edge in template_mesh.Elements1D():
        v1, v2 = edge.vertices
        newmesh.Add(Element1D([v1,v2],index=regions[edge.edgenr-1]))

        #Find periodic boundary if applied
        if edge.edgenr in idx_xm:
            for i in edge.vertices:
                if i not in master: master.append(i)
        if edge.edgenr in idx_xs:
            for i in edge.vertices:
                if i not in slave: slave.append(i)

    for i,j in zip(master,slave):
        newmesh.AddPointIdentification(i,j,1,2)
    return ngsolve.Mesh(newmesh)

def tang(u,n):
    return u-(u*n)*n

mesh_list = {"periodic" : CreateSquareMesh,
             "cavity"   : CreateSquareMesh,
             "unit": CreateSquareMesh,
             "cylinder" : CreateCylinderMesh}