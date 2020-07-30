from multiprocessing import Pool
import  multiprocessing
import os
import numpy as np
import copy as cp
import inp
import MP2
import trans_mo
import diagrams
import cc_update_parallel
import cc_symmetrize
import time
#from memory_profiler import profile

mol = inp.mol
# Obtain the number of atomic orbitals in the basis set
nao = MP2.nao
start = time.time()

# import important stuff
E_hf = trans_mo.E_hf
Fock_mo = MP2.Fock_mo
twoelecint_mo = MP2.twoelecint_mo
t1 = MP2.t1
D1 = MP2.D1
t2 = MP2.t2
D2=MP2.D2
So = MP2.So
Do=MP2.Do
Sv = MP2.Sv
Dv=MP2.Dv
occ = MP2.occ
virt = MP2.virt
E_old = MP2.E_mp2_tot
n_iter = inp.n_iter
calc = inp.calc
conv = 10**(-inp.conv)
max_diis = inp.max_diis
o_act = inp.o_act
v_act = inp.v_act
start = time.time()

#    Evaluate the energy
def energy_ccd(t2):
  E_ccd = 2*np.einsum('ijab,ijab',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao]) - np.einsum('ijab,ijba',t2,twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
  return E_ccd

def energy_ccsd(t1,t2):
  E_ccd = energy_ccd(t2)
  E_ccd += 2*np.einsum('ijab,ia,jb',twoelecint_mo[:occ,:occ,occ:nao,occ:nao],t1,t1) - np.einsum('ijab,ib,ja',twoelecint_mo[:occ,:occ,occ:nao,occ:nao],t1,t1)
  return E_ccd

def convergence_I(E_ccd,E_old,eps_t,eps_So,eps_Sv):
  del_E = E_ccd - E_old
  if abs(eps_t) <= conv and abs(eps_So) <= conv and abs(eps_Sv) <= conv and abs(del_E) <= conv:
    print "ccd converged!!!"
    print "Total energy is : "+str(E_hf + E_ccd)
    return True
  else:
    print "cycle number : "+str(x+1)
    print "change in t1+t2 , So, Sv : "+str(eps_t)+" "+str(eps_So)+" "+str(eps_Sv)
    print "energy difference : "+str(del_E)
    print "energy : "+str(E_hf + E_ccd)
    E_old = E_ccd
    return False

def convergence(E_ccd,E_old,eps):
  del_E = E_ccd - E_old
  if abs(eps) <= conv and abs(del_E) <= conv:
    print "ccd converged!!!"
    print "Total energy is : "+str(E_hf + E_ccd)
    return True
  else:
    print "cycle number : "+str(x+1)
    print "change in t1 and t2 : "+str(eps)
    print "energy difference : "+str(del_E)
    print "energy : "+str(E_hf + E_ccd)
    return False

for x in range(0,n_iter):
  if calc == 'ICCSD':
    pool=Pool(12)
    print "----------ICCSD------------"
    tau = cp.deepcopy(t2)
    tau += np.einsum('ia,jb->ijab',t1,t1)
    
    II_oo = diagrams.So_int_diagrams(So,t2,t1)[2]
    II_vv = diagrams.Sv_int_diagrams(Sv,t2,t1)[2]

    result_comb_temp1 = pool.apply_async(diagrams.update1,args=(t1,t2,tau,))
    result_comb_temp2 = pool.apply_async(diagrams.update2,args=(t1,tau,))
    result_comb_temp3 = pool.apply_async(diagrams.update10,args=(t1,t2,))
    R_ijab3_temp = pool.apply_async(diagrams.update3,args=(tau,t1,t2,))
    R_ijab4_temp = pool.apply_async(diagrams.update4,args=(t1,t2,))  
    R_ijab5_temp = pool.apply_async(diagrams.update5,args=(t1,t2,))  
    R_ijab6_temp = pool.apply_async(diagrams.update6,args=(t1,t2,))  
    R_ijab7_temp = pool.apply_async(diagrams.update7,args=(t1,t2,))  
    R_ijab8_temp = pool.apply_async(diagrams.update8,args=(t1,t2,))  
    R_ijab9_temp = pool.apply_async(diagrams.update9,args=(tau,))  
    R_iuab_temp = pool.apply_async(diagrams.Sv_diagrams,args=(Sv,t1,t2,II_vv,))  
    R_ijav_temp = pool.apply_async(diagrams.So_diagrams,args=(So,t1,t2,II_oo,))  
   
    pool.close()
    pool.join()

    R_ia1, R_ijab1 = result_comb_temp1.get() 
    R_ia2, R_ijab2 = result_comb_temp2.get() 
    R_ia10, R_ijab10 = result_comb_temp3.get() 
    R_ijab3 = R_ijab3_temp.get()
    R_ijab4 = R_ijab4_temp.get()
    R_ijab5 = R_ijab5_temp.get()
    R_ijab6 = R_ijab6_temp.get()
    R_ijab7 = R_ijab7_temp.get()
    R_ijab8 = R_ijab8_temp.get()
    R_ijab9 = R_ijab9_temp.get()

    R_ia = (R_ia1+R_ia2+R_ia10)
    R_ia += diagrams.So_int_diagrams(So,t2,t1)[1] 
    R_ia += diagrams.Sv_int_diagrams(Sv,t2,t1)[1] 
    R_ijab = (R_ijab1+R_ijab2+R_ijab3+R_ijab4+R_ijab5+R_ijab6+R_ijab7+R_ijab8+R_ijab9+R_ijab10)
    R_ijab += diagrams.So_int_diagrams(So,t2,t1)[0]
    R_ijab += diagrams.Sv_int_diagrams(Sv,t2,t1)[0]
    R_ijab = cc_symmetrize.symmetrize(R_ijab)

    R_iuab = R_iuab_temp.get()
    R_iuab += diagrams.T1_contribution_Sv(t1)
    R_iuab += diagrams.coupling_terms_So(So,t2)[0]
    R_iuab += diagrams.w2_int_2(So,Sv,t2)
    R_ijav = R_ijav_temp.get() 
    R_ijav += diagrams.T1_contribution_So(t1)
    R_ijav += diagrams.coupling_terms_Sv(Sv,t2)[0]
    R_ijav += diagrams.w2_int_1(So,Sv,t2)
 
    oldt2 = t2.copy() 
    oldt1 = t1.copy() 
    oldSo = So.copy() 
    oldSv = Sv.copy() 
    

    eps_t, t1, t2 = cc_update_parallel.update_t1t2(R_ia,R_ijab,t1,t2)
    eps_So, So = cc_update_parallel.update_So(R_ijav,So)
    eps_Sv, Sv = cc_update_parallel.update_Sv(R_iuab,Sv)
       
    E_ccd = energy_ccsd(t1,t2)
    val = convergence_I(E_ccd,E_old,eps_t,eps_So,eps_Sv)
    if val == True :
        break
    else:
        E_old = E_ccd


  if calc == 'ICCSD-PT':
    pool=Pool(12)
    print "----------ICCSD-PT------------"
    tau = cp.deepcopy(t2)
    tau += np.einsum('ia,jb->ijab',t1,t1)
    
    II_oo = diagrams.So_int_diagrams(So,t2)[1]
    II_vv = diagrams.Sv_int_diagrams(Sv,t2)[1]

    result_comb_temp1 = pool.apply_async(diagrams.update1,args=(t1,t2,tau,))
    result_comb_temp2 = pool.apply_async(diagrams.update2,args=(t1,tau,))
    result_comb_temp3 = pool.apply_async(diagrams.update10,args=(t1,t2,))
    R_ijab3_temp = pool.apply_async(diagrams.update3,args=(tau,t1,t2,))
    R_ijab4_temp = pool.apply_async(diagrams.update4,args=(t1,t2,))  
    R_ijab5_temp = pool.apply_async(diagrams.update5,args=(t1,t2,))  
    R_ijab6_temp = pool.apply_async(diagrams.update6,args=(t1,t2,))  
    R_ijab7_temp = pool.apply_async(diagrams.update7,args=(t1,t2,))  
    R_ijab8_temp = pool.apply_async(diagrams.update8,args=(t1,t2,))  
    R_ijab9_temp = pool.apply_async(diagrams.update9,args=(tau,))  

    pool.close()
    pool.join()

    R_ia1, R_ijab1 = result_comb_temp1.get() 
    R_ia2, R_ijab2 = result_comb_temp2.get() 
    R_ia10, R_ijab10 = result_comb_temp3.get() 
    R_ijab3 = R_ijab3_temp.get()
    R_ijab4 = R_ijab4_temp.get()
    R_ijab5 = R_ijab5_temp.get()
    R_ijab6 = R_ijab6_temp.get()
    R_ijab7 = R_ijab7_temp.get()
    R_ijab8 = R_ijab8_temp.get()
    R_ijab9 = R_ijab9_temp.get()

    R_ia = (R_ia1+R_ia2+R_ia10)
    R_ijab = (R_ijab1+R_ijab2+R_ijab3+R_ijab4+R_ijab5+R_ijab6+R_ijab7+R_ijab8+R_ijab9+R_ijab10)
    R_ijab += diagrams.So_int_diagrams(So,t2)[0]
    R_ijab += diagrams.Sv_int_diagrams(Sv,t2)[0]
    R_ijab = cc_symmetrize.symmetrize(R_ijab)

    oldt2 = t2.copy() 
    oldt1 = t1.copy() 

    eps_t, t1, t2 = cc_update_parallel.update_t1t2(R_ia,R_ijab,t1,t2)
    
    E_ccd = energy_ccsd(t1,t2)
    val = convergence(E_ccd,E_old,eps_t)
    if val == True :
        break
    else:
        E_old = E_ccd

end = time.time()
print "parallel overall time",(end - start)
