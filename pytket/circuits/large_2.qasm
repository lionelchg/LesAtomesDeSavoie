OPENQASM 2.0;
include "qelib1.inc";

qreg q[8];
creg c[8];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
cz q[7],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[5];
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
cz q[1],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(0.441275860354688*pi) q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[6],q[4];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[1],q[6];
rx(0.5*pi) q[4];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[3],q[1];
cz q[4],q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
cz q[5],q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
cz q[0],q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
cz q[7],q[0];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[5];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
rx(0.5*pi) q[5];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rx(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
cz q[2],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(1.9749744848290767*pi) q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[6],q[0];
rz(0.5*pi) q[0];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
cz q[1],q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
cz q[3],q[1];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rx(0.5*pi) q[1];
cz q[7],q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[7];
cz q[2],q[7];
rz(0.5*pi) q[3];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[7];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[7];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
cz q[4],q[2];
rz(0.5*pi) q[7];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[7];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
cz q[7],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
rz(0.26471561722784087*pi) q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[5],q[6];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[1],q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[2],q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
cz q[7],q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
rx(0.5*pi) q[2];
cz q[2],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(0.11300130629634864*pi) q[4];
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
cz q[7],q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rx(0.5*pi) q[4];
cz q[5],q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rx(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
cz q[1],q[5];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
cz q[3],q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[1];
cz q[6],q[3];
rx(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
cz q[2],q[6];
rx(0.5*pi) q[2];
rz(0.5*pi) q[6];
cz q[2],q[0];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[6];
cz q[0],q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[5],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
rz(1.1182495558100667*pi) q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[6],q[1];
rz(0.5*pi) q[1];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
cz q[4],q[6];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
cz q[5],q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
cz q[3],q[1];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[0],q[5];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
cz q[2],q[0];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[1],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(0.7809451839554713*pi) q[4];
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
cz q[6],q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
cz q[5],q[6];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[0],q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
cz q[4],q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
cz q[1],q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[2],q[6];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
cz q[3],q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
cz q[6],q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(1.4024611497792667*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
cz q[6],q[5];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[2],q[6];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[4],q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
cz q[6],q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
cz q[3],q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
cz q[0],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(1.1163563512834829*pi) q[4];
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
cz q[3],q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
cz q[6],q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
cz q[4],q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[6],q[3];
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
cz q[2],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
rz(1.3681855906617035*pi) q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[7],q[5];
rz(0.5*pi) q[5];
rz(0.5*pi) q[7];
rx(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[5];
rz(0.5*pi) q[7];
cz q[2],q[7];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[7];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[7];
cz q[0],q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
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
cz q[6],q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[3];
cz q[4],q[6];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[3],q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[6];
cz q[0],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
rz(1.5410334645292878*pi) q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[4],q[1];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
cz q[7],q[4];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rx(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
cz q[6],q[7];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[5],q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
cz q[0],q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[3],q[0];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[0];
cz q[4],q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
rz(0.7738367406482032*pi) q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[1],q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[7];
cz q[2],q[1];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
cz q[6],q[2];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
cz q[0],q[6];
rx(0.5*pi) q[2];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[6];
cz q[3],q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
cz q[4],q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[0],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rz(1.3865759293761617*pi) q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[5],q[2];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
cz q[6],q[5];
rx(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[5];
cz q[7],q[6];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[3],q[7];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[0],q[3];
rx(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
cz q[2],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
rz(1.9895882444007318*pi) q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[4],q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
rx(0.5*pi) q[4];
rx(0.5*pi) q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
cz q[6],q[4];
rx(0.5*pi) q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[4];
cz q[5],q[6];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
cz q[3],q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
cz q[2],q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
cz q[4],q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
rz(0.5569637487958603*pi) q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[5],q[1];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[1];
cz q[6],q[5];
rx(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
cz q[2],q[6];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
cz q[4],q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
cz q[7],q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
cz q[4],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
rz(1.4084718634894948*pi) q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[3],q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
cz q[1],q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
cz q[4],q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
cz q[7],q[4];
rx(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[4];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[5],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
rz(0.5893782356857833*pi) q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[3],q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
cz q[2],q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
cz q[0],q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rx(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
cz q[4],q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[0];
cz q[3],q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
cz q[7],q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
cz q[1],q[6];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[4];
cz q[5],q[7];
rx(0.5*pi) q[6];
rz(0.5*pi) q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rz(0.5*pi) q[7];
cz q[6],q[4];
rx(0.5*pi) q[5];
rz(0.5*pi) q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[5];
rx(0.5*pi) q[7];
rx(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[4];
rz(0.09574165318255701*pi) q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[6],q[4];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[1],q[6];
rx(0.5*pi) q[4];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[3],q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[3];
cz q[7],q[3];
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
cz q[1],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
rz(1.3307103169238674*pi) q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[1],q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[5];
cz q[2],q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[5];
rz(0.5*pi) q[1];
cz q[3],q[2];
rz(0.5*pi) q[5];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[2];
cz q[7],q[3];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[7];
rz(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[7];
rx(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[7];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[7];
cz q[0],q[7];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[7];
rz(0.5*pi) q[3];
rx(0.5*pi) q[7];
rx(0.5*pi) q[3];
rz(0.5*pi) q[7];
rz(0.5*pi) q[3];
cz q[7],q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
cz q[4],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
rz(0.7646489011882271*pi) q[2];
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
cz q[6],q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
cz q[4],q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[7],q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[0],q[7];
rx(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rx(0.5*pi) q[7];
cz q[0],q[2];
rz(0.5*pi) q[7];
rz(0.5*pi) q[2];
rx(0.5*pi) q[7];
rx(0.5*pi) q[2];
rz(0.5*pi) q[7];
rz(0.5*pi) q[2];
cz q[2],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
rz(1.1460292909964684*pi) q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[6],q[7];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[4],q[6];
rz(0.5*pi) q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[2],q[4];
rx(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[0],q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[2];
rz(0.5*pi) q[0];
rx(0.5*pi) q[2];
rx(0.5*pi) q[0];
rx(0.5*pi) q[2];
rz(0.5*pi) q[0];
cz q[2],q[1];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
rx(0.5*pi) q[0];
rx(0.5*pi) q[1];
rz(0.5*pi) q[0];
rz(0.5*pi) q[1];
cz q[1],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[3],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(0.8968382326431712*pi) q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[5],q[0];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
cz q[3],q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
cz q[6],q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
cz q[7],q[6];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[1],q[7];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[2],q[1];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
cz q[6],q[2];
rz(0.5*pi) q[7];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(0.6432735189809691*pi) q[0];
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
rz(0.5*pi) q[0];
cz q[7],q[4];
rx(0.5*pi) q[0];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[0];
rx(0.5*pi) q[4];
rx(0.5*pi) q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
cz q[1],q[7];
rz(0.5*pi) q[4];
rz(0.5*pi) q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[4];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[4];
rz(0.5*pi) q[7];
cz q[2],q[1];
rx(0.5*pi) q[4];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rx(0.5*pi) q[7];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[7];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[7];
rx(0.5*pi) q[1];
cz q[6],q[2];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[6];
cz q[3],q[2];
rz(0.5*pi) q[2];
rx(0.5*pi) q[2];
rz(0.5*pi) q[2];
cz q[2],q[1];
rz(0.5*pi) q[1];
rx(0.5*pi) q[1];
rz(0.5*pi) q[1];
cz q[1],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[7];
rz(0.5*pi) q[7];
rx(0.5*pi) q[7];
rz(0.5*pi) q[7];
cz q[7],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
rz(0.2513784374456445*pi) q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[7],q[5];
rz(0.5*pi) q[5];
rz(0.5*pi) q[7];
rx(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[5];
rz(0.5*pi) q[7];
rx(0.5*pi) q[5];
cz q[6],q[7];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
rx(0.5*pi) q[5];
rx(0.5*pi) q[6];
rx(0.5*pi) q[7];
rz(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[7];
cz q[1],q[6];
rz(0.5*pi) q[5];
rx(0.5*pi) q[7];
rz(0.5*pi) q[1];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[1];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[6];
cz q[2],q[1];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[6];
rx(0.5*pi) q[1];
rx(0.5*pi) q[2];
rz(0.5*pi) q[6];
rz(0.5*pi) q[1];
rz(0.5*pi) q[2];
rx(0.5*pi) q[6];
rx(0.5*pi) q[1];
cz q[3],q[2];
rz(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[6];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
rx(0.5*pi) q[2];
rx(0.5*pi) q[3];
rz(0.5*pi) q[2];
rz(0.5*pi) q[3];
cz q[2],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[0],q[5];
rz(0.5*pi) q[5];
rx(0.5*pi) q[5];
rz(0.5*pi) q[5];
cz q[5],q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
rz(0.4235415248615433*pi) q[3];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
rz(0.5*pi) q[3];
cz q[5],q[3];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
cz q[0],q[5];
rz(0.5*pi) q[3];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rz(0.5*pi) q[5];
cz q[6],q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[5];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[5];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
cz q[5],q[3];
cz q[4],q[6];
rx(0.5*pi) q[0];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[0];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[2],q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[2];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[2];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[3],q[4];
rz(0.5*pi) q[4];
rx(0.5*pi) q[4];
rz(0.5*pi) q[4];
cz q[4],q[6];
rz(0.5*pi) q[6];
rx(0.5*pi) q[6];
rz(0.5*pi) q[6];
cz q[6],q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
rz(0.09063090113245265*pi) q[0];
rz(0.5*pi) q[0];
rx(0.5*pi) q[0];
rz(0.5*pi) q[0];
cz q[6],q[0];
rz(0.5*pi) q[0];
rz(0.5*pi) q[6];
rx(0.5*pi) q[0];
rx(0.5*pi) q[6];
rz(0.5*pi) q[0];
rz(0.5*pi) q[6];
cz q[4],q[6];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
rx(0.5*pi) q[4];
rx(0.5*pi) q[6];
rz(0.5*pi) q[4];
rz(0.5*pi) q[6];
cz q[3],q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
rx(0.5*pi) q[3];
rx(0.5*pi) q[4];
rz(0.5*pi) q[3];
rz(0.5*pi) q[4];
cz q[5],q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[3];
rx(0.5*pi) q[4];
rx(0.5*pi) q[5];
rx(0.5*pi) q[3];
rz(0.5*pi) q[4];
rz(0.5*pi) q[3];
rx(0.5*pi) q[3];
