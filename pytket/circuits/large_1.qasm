OPENQASM 2.0;
include "qelib1.inc";

qreg q[6];
creg c[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
cz q[3],q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
cz q[4],q[1];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[0],q[2];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.30220847759223024*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[1];
rz(1.0310570383957178*pi) q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
cz q[4],q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
cz q[0],q[2];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
cz q[3],q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
cz q[1],q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(1.5250286058865397*pi) q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[2],q[0];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[0];
cz q[4],q[2];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
cz q[5],q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
cz q[3],q[5];
rz(0.5*pi) q[4];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
cz q[1],q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[3];
cz q[4],q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
cz q[5],q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
cz q[0],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
rz(1.138725968935714*pi) q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[1],q[3];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
cz q[0],q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
cz q[5],q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
cz q[4],q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[3],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rz(0.46047641342990975*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[1],q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
cz q[0],q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[4],q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
cz q[5],q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[3],q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
cz q[1],q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(0.8573171829576256*pi) q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[2],q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
cz q[0],q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
cz q[1],q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[0];
cz q[5],q[1];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[3],q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rz(1.293634096624129*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[1],q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
cz q[4],q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
cz q[5],q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
cz q[0],q[5];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
cz q[3],q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[1],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rz(0.15565531058409876*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[4],q[2];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[2];
cz q[5],q[4];
rx(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
cz q[0],q[5];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
cz q[3],q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
cz q[1],q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
cz q[1],q[5];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
cz q[5],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(0.4485622732873449*pi) q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[0],q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
cz q[5],q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
cz q[1],q[5];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[5];
rx(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[3],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(0.7429757100153136*pi) q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[4],q[0];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
cz q[5],q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
cz q[3],q[5];
rx(0.5*pi) q[4];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[2],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
rz(0.42654451945601735*pi) q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[3],q[1];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
cz q[5],q[3];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
cz q[2],q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[4],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rz(0.803943921940065*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[0],q[2];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
cz q[3],q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[0];
cz q[5],q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
cz q[4],q[5];
cz q[1],q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[0],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
rz(1.2625888946619912*pi) q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[4],q[3];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
cz q[2],q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
cz q[5],q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
cz q[0],q[5];
rx(0.5*pi) q[2];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
cz q[1],q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[4],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rz(1.4251191959229599*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[3],q[2];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
cz q[0],q[3];
rz(0.5*pi) q[2];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
cz q[4],q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
cz q[1],q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.645376221663335*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[1],q[2];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[1],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(1.8666790271677978*pi) q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[3],q[0];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[0];
cz q[4],q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
cz q[2],q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
cz q[1],q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[1];
cz q[2],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(0.8614838179995494*pi) q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[1],q[4];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
cz q[0],q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[5],q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[2],q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
cz q[4],q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
rz(0.6336176664683306*pi) q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[0],q[1];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[5],q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
cz q[2],q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
cz q[4],q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
cz q[0],q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
cz q[5],q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
rz(1.303653995972691*pi) q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[2],q[1];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
cz q[3],q[2];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[2];
cz q[4],q[3];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[3];
cz q[5],q[4];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
cz q[0],q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];