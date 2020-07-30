
                   ##----------------------------------------------------------------------------------------------------------------##
                                   
                         # Construction of intermediate and diagrams for the parallel calculation of the ground state energy #
                                           # Author: Soumi Tribedi, Anish Chakraborty & Rahul Maitra #
                                                           # Date - 10th Dec, 2019 # 

                   ##----------------------------------------------------------------------------------------------------------------##

##--------------------------------------------## 
          #Import important modules#
##--------------------------------------------## 

from multiprocessing import pool
import gc
import numpy as np
import copy as cp
import MP2
import inp

mol = inp.mol

##------------------------------------------------##
           #Import important parameters#
##------------------------------------------------##

twoelecint_mo = MP2.twoelecint_mo 
Fock_mo = MP2.Fock_mo
t1 = MP2.t1
t2 = MP2.t2
So = MP2.So
Sv = MP2.Sv
occ = MP2.occ
virt = MP2.virt
nao = MP2.nao
 
##-------------------------------------------------------------##
          #Active orbital imported from input file#
##-------------------------------------------------------------##

o_act = inp.o_act
v_act = inp.v_act
act = o_act + v_act


                     ##--------------------------------------------------------------------------------------------------------------------------------------------------##
                                                                           ##Construction of intermediates and diagrams##
                     ##--------------------------------------------------------------------------------------------------------------------------------------------------##
                              

def update1(t1,t2,tau):
  I_vv = cp.deepcopy(Fock_mo[occ:nao,occ:nao])
  I_vv += -2*np.einsum('cdkl,klad->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdkl,klda->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) #int for diag k,i(one-body), 24,26(twobody)

  I_oo = cp.deepcopy(Fock_mo[:occ,:occ])
  I_oo += 2*np.einsum('cdkl,ilcd->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],tau) - np.einsum('dckl,lidc->ik',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],tau) #int diag l,j,m,n(one-body) & 25,27,35,38'(2body)
 
  R_ia1 = cp.deepcopy(Fock_mo[:occ,occ:nao]) 
  R_ia1 += -np.einsum('ik,ka->ia',I_oo,t1)                                                  #diagrams linear 1 & non-linear l,j,m,n
  R_ia1 += np.einsum('ca,ic->ia',I_vv,t1)                                                   #diagrams linear 2 & non-linear k,i
  
  I_vv += 2*np.einsum('bcja,jb->ca',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1)         #intermediate for diagram non-linear 6
  I_vv += -np.einsum('cbja,jb->ca',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t1)          #intermediate for diagram non-linear 7
  I_vv += -2*np.einsum('dclk,ld,ka->ca',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)     #intermediate for diagram non-linear 34'
  I_oo += 2*np.einsum('ibkj,jb->ik',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)               #intermediate for diagram non-linear 5
  I_oo += -np.einsum('ibjk,jb->ik',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)                #intermediate for diagram non-linear 8
  
  R_ijab1 = 0.5*cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ:nao])
  R_ijab1 += -np.einsum('ik,kjab->ijab',I_oo,t2)                                            #diagrams linear 1 & non-linear 25,27,5,8,35,38'
  R_ijab1 += np.einsum('ca,ijcb->ijab',I_vv,t2)                                             #diagrams linear 2 & non-linear 24,26,34',6,7

  return R_ia1,R_ijab1
  I_oo = None
  I_vv = None
  R_ia1 = None
  R_ijab1 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update2(t1,tau):
  R_ia2 = -2*np.einsum('ibkj,kjab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],tau)            #diagrams linear 5 & non-linear a
  R_ia2 += np.einsum('ibkj,jkab->ia',twoelecint_mo[:occ,occ:nao,:occ,:occ],tau)              #diagrams linear 6 & non-linear b
  R_ia2 += 2*np.einsum('cdak,ikcd->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],tau)      #diagrams linear 7 & non-linear c
  R_ia2 += -np.einsum('cdak,ikdc->ia',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],tau)       #diagrams linear 8 & non-linear d
  R_ia2 += 2*np.einsum('icak,kc->ia',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1)            #diagram  linear 3
  R_ia2 += -np.einsum('icka,kc->ia',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1)             #diagram  linear 4
  
  R_ijab2 = -np.einsum('ickb,ka,jc->ijab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],t1,t1)    #diagrams non-linear 3
  R_ijab2 += -np.einsum('icak,jc,kb->ijab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],t1,t1)   #diagrams non-linear 4
  R_ijab2 += -np.einsum('ijkb,ka->ijab',twoelecint_mo[:occ,:occ,:occ,occ:nao],t1)            #diagram linear 3
  R_ijab2 += np.einsum('cjab,ic->ijab',twoelecint_mo[occ:nao,:occ,occ:nao,occ:nao],t1)       #diagram linear 4

  return R_ia2,R_ijab2
  R_ia2 = None
  R_ijab2 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update3(tau,t1,t2):
  Ioooo = cp.deepcopy(twoelecint_mo[:occ,:occ,:occ,:occ])
  Ioooo += np.einsum('cdkl,ijcd->ijkl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)          #intermediate for diagram non-linear 38
  Ioooo_2 = 0.5*np.einsum('cdkl,ic,jd->ijkl',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1) #intermediate for diagram non-linear 37
  
  R_ijab3 = 0.5*np.einsum('ijkl,klab->ijab',Ioooo,tau)                                       #diagrams linear 9 & non-linear 1,22,38
  R_ijab3 += np.einsum('ijkl,klab->ijab',Ioooo_2,t2)                                         #diagram  non-linear 37

  return R_ijab3
  I_oooo = None
  I_vvvv = None
  R_ijab3 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update4(t1,t2):
  Iovov = cp.deepcopy(twoelecint_mo[:occ,occ:nao,:occ,occ:nao])
  Iovov += -0.5*np.einsum('dckl,ildb->ickb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)      #intermediate for diagram non-linear 23
  Iovov_2 = cp.deepcopy(twoelecint_mo[:occ,occ:nao,:occ,occ:nao])                             #intermediate for diagram linear 7
  Iovov_3 = -np.einsum('dckl,ildb->ickb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)         #intermediate for diagrams 36
  
  R_ijab4 = - np.einsum('ickb,kjac->ijab',Iovov,t2)                                           #diagrams linear 10 & non-linear 23 
  R_ijab4 += -np.einsum('icka,kjcb->ijab',Iovov_2,t2)                                         #diagram  linear 7 
  R_ijab4 += -np.einsum('ickb,jc,ka->ijab',Iovov_3,t1,t1)                                     #diagrams non-linear 36

  return R_ijab4
  Iovov = None
  Iovov_2 = None
  Iovov_3 = None
  R_ijab4 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update5(t1,t2):
  I_oovo = np.zeros((occ,occ,virt,occ))
  I_oovo += -np.einsum('cikl,jlca->ijak',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2)            #intermediate for diagram non-linear 11
  I_oovo += np.einsum('cdka,jicd->ijak',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)       #intermediate for diagram non-linear 12
  I_oovo += -np.einsum('jclk,lica->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)            #intermediate for diagram non-linear 13
  I_oovo += 2*np.einsum('jckl,ilac->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)           #intermediate for diagram non-linear 15
  I_oovo += -np.einsum('jckl,ilca->ijak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t2)            #intermediate for diagram non-linear 17
   
  R_ijab5 = -np.einsum('ijak,kb->ijab',I_oovo,t1)                                             #diagrams non-linear 11,12,13,15,17

  return R_ijab5
  I_oovo = None
  R_ijab5 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update6(t1,t2):
  I_vovv = np.zeros((virt,occ,virt,virt))
  I_vovv += np.einsum('cjkl,klab->cjab',twoelecint_mo[occ:nao,:occ,:occ,:occ],t2)              #intermediate for diagram non-linear 9
  I_vovv += -np.einsum('cdlb,ljad->cjab',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)       #intermediate for diagram non-linear 10
  I_vovv += -np.einsum('cdka,kjdb->cjab',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],t2)       #intermediate for diagram non-linear 14
  I_vovv += 2*np.einsum('cdal,ljdb->cjab',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2)      #intermediate for diagram non-linear 16
  I_vovv += -np.einsum('cdal,jldb->cjab',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t2)       #intermediate for diagram non-linear 18

  R_ijab6 = np.einsum('cjab,ic->ijab',I_vovv,t1)                                               #diagrams non-linear 9,10,14,16,18

  return R_ijab6
  I_vovv = None
  R_ijab6 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update7(t1,t2):
  Iovvo = cp.deepcopy(twoelecint_mo[:occ,occ:nao,occ:nao,:occ])
  Iovvo += np.einsum('dclk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - 0.5*np.einsum('cdlk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)                                                                    #intermediates for diagrams non-linear 19,28,20 

  Iovvo_2 = cp.deepcopy(twoelecint_mo[:occ,occ:nao,occ:nao,:occ])
  Iovvo_2 += -0.5*np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)  - np.einsum('dckl,ljdb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) #int for diagrams non-linear 21,29
  
  Iovvo_3 = 2*np.einsum('dclk,jlbd->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) - np.einsum('dclk,jldb->jcbk',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2) + np.einsum('cdak,ic->idak',twoelecint_mo[occ:nao,occ:nao,occ:nao,:occ],t1)                                                                    #intermediates for diagrams non-linear 32,33,31
  Iovvo_3 += -np.einsum('iclk,la->icak',twoelecint_mo[:occ,occ:nao,:occ,:occ],t1)             #intermediate  for diagram  non-linear 30
  
  R_ijab7 = 2*np.einsum('jcbk,kica->ijab',Iovvo,t2)                                           #diagrams linear 6 & non-linear 19,28,20 
  R_ijab7 += - np.einsum('jcbk,ikca->ijab',Iovvo_2,t2)                                        #diagrams linear 8 & non-linear 21,29
  R_ijab7 += -np.einsum('jcbk,ic,ka->ijab',Iovvo_3,t1,t1)                                     #diagrams non-linear 32,33,31,30

  return R_ijab7
  I_ovvo = None
  I_ovvo_2 = None
  I_ovvo_3 = None
  R_ijab7 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update8(t1,t2): 
  I_voov = -np.einsum('cdkl,kjdb->cjlb',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t2)          #intermediate for diagram non-linear 39
  R_ijab8 = -np.einsum('cjlb,ic,la->ijab',I_voov,t1,t1)                                       #diagram non-linear 39

  return R_ijab8
  I_voov = None
  R_ijab8 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update9(tau):
  Ivvvv = cp.deepcopy(twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao])                        #intermediates for diagrams linear 5 & non-linear 2
  R_ijab9 = 0.5*np.einsum('cdab,ijcd->ijab',Ivvvv,tau)                                       #diagrams linear 5 & non-linear 2

  return R_ijab9
  Ivvvv = None
  R_ijab9 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def update10(t1,t2):
  I1 = 2*np.einsum('cbkj,kc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1)                #intermediates for diagrams linear e,f
  I2 = -np.einsum('cbjk,kc->bj',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1)                 #intermediates for diagrams linear g,h
  I3 = -np.einsum('cdkl,ic,ka->idal',twoelecint_mo[occ:nao,occ:nao,:occ,:occ],t1,t1)         #intermediate  for diagram non-linear 40
  Iooov = np.einsum('dl,ijdb->ijlb',I2,t2)                                                   #intermediates for diagrams non-linear34,30
 
  R_ia10 = 2*np.einsum('bj,ijab->ia',I1,t2) - np.einsum('bj,ijba->ia',I1,t2)                 #diagrams linear one-body e,f
  R_ia10 += 2*np.einsum('bj,ijab->ia',I2,t2) - np.einsum('bj,ijba->ia',I2,t2)                #diagrams linear one-body g,h

  R_ijab10 = -np.einsum('ijlb,la->ijab',Iooov,t1)                                            #diagram non-linear 34,30
  R_ijab10 += -0.5*np.einsum('idal,jd,lb->ijab',I3,t1,t1)                                    #diagram non-linear 40

  return R_ia10,R_ijab10
  I1 = None
  I2 = None
  I3 = None
  Iooov = None
  R_ia10 = None
  R_ijab10 = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def So_int_diagrams(So,t2,t1):
  II_oo = np.zeros((occ,occ)) 
  II_oo[:,occ-o_act:occ] += -2*0.25*np.einsum('ciml,mlcv->iv',twoelecint_mo[occ:nao,:occ,:occ,:occ],So) + 0.25*np.einsum('diml,lmdv->iv',twoelecint_mo[occ:nao,:occ,:occ,:occ],So)
  
  R_ia = -np.einsum('ik,ka->ia',II_oo,t1)
  R_ijab = -np.einsum('ik,kjab->ijab',II_oo,t2)   

  return R_ijab,R_ia,II_oo
  II_oo = None
  R_ijab = None
  R_ia = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def Sv_int_diagrams(Sv,t2,t1):
  II_vv = np.zeros((virt,virt))
  II_vv[:v_act,:] += 2*0.25*np.einsum('dema,mude->ua',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],Sv) - 0.25*np.einsum('dema,mued->ua',twoelecint_mo[occ:nao,occ:nao,:occ,occ:nao],Sv)
 
  R_ia = np.einsum('ca,ic->ia',II_vv,t1)
  R_ijab = np.einsum('ca,ijcb->ijab',II_vv,t2)   

  return R_ijab,R_ia,II_vv
  II_vv = None
  R_ijab = None
  R_ia = None
  gc.collect()
 
##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def Sv_diagrams(Sv,t1,t2,II_vv):
  R_iuab = cp.deepcopy(twoelecint_mo[:occ,occ:occ+v_act,occ:nao,occ:nao])
  R_iuab += -np.einsum('ik,kuab->iuab',Fock_mo[:occ,:occ],Sv)
  R_iuab += np.einsum('da,iudb->iuab',Fock_mo[occ:nao,occ:nao],Sv)
  R_iuab += np.einsum('db,iuad->iuab',Fock_mo[occ:nao,occ:nao],Sv)
  R_iuab += np.einsum('edab,iued->iuab',twoelecint_mo[occ:nao,occ:nao,occ:nao,occ:nao],Sv)
  R_iuab += 2*np.einsum('dukb,kida->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
  R_iuab += 2*np.einsum('idak,kudb->iuab',twoelecint_mo[:occ,occ:nao,occ:nao,:occ],Sv)
  R_iuab += -np.einsum('idka,kudb->iuab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],Sv)
  R_iuab += -np.einsum('udkb,kida->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
  R_iuab += -np.einsum('dukb,kiad->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,occ:nao],t2)
  R_iuab += -np.einsum('dika,kubd->iuab',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],Sv)
  R_iuab += np.einsum('uikl,klba->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,:occ],t2)
  R_iuab += -np.einsum('udka,kibd->iuab',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,occ:nao],t2)
  R_iuab += -np.einsum('idkb,kuad->iuab',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],Sv)

  return R_iuab
  R_iuab = None
  II_vv = None   
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def So_diagrams(So,t1,t2,II_oo):
  R_ijav = cp.deepcopy(twoelecint_mo[:occ,:occ,occ:nao,occ-o_act:occ])
  R_ijav += np.einsum('da,ijdv->ijav',Fock_mo[occ:nao,occ:nao],So)
  R_ijav += -np.einsum('jl,ilav->ijav',Fock_mo[:occ,:occ],So)
  R_ijav += -np.einsum('il,ljav->ijav',Fock_mo[:occ,:occ],So)
  R_ijav += 2*np.einsum('djlv,lida->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
  R_ijav += 2*np.einsum('dila,ljdv->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],So)
  R_ijav += -np.einsum('djlv,liad->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ-o_act:occ],t2)
  R_ijav += -np.einsum('dila,jldv->ijav',twoelecint_mo[occ:nao,:occ,:occ,occ:nao],So)
  R_ijav += -np.einsum('dial,ljdv->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,:occ],So)
  R_ijav += -np.einsum('djvl,lida->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,:occ],t2)
  R_ijav += np.einsum('ijlm,lmav->ijav',twoelecint_mo[:occ,:occ,:occ,:occ],So)
  R_ijav += np.einsum('cdva,jicd->ijav',twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,occ:nao],t2)
  R_ijav += -np.einsum('jdla,ildv->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ:nao],So)
  R_ijav += -np.einsum('idlv,ljad->ijav',twoelecint_mo[:occ,occ:nao,:occ,occ-o_act:occ],t2)

  return R_ijav
  R_ijav = None
  II_oo = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def T1_contribution_Sv(t1):
  R_iuab = -np.einsum('uika,kb->iuab',twoelecint_mo[occ:occ+v_act,:occ,:occ,occ:nao],t1)
  R_iuab += np.einsum('duab,id->iuab',twoelecint_mo[occ:nao,occ:occ+v_act,occ:nao,occ:nao],t1)
  R_iuab += -np.einsum('iukb,ka->iuab',twoelecint_mo[:occ,occ:occ+v_act,:occ,occ:nao],t1)

  return R_iuab
  R_iuab = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def T1_contribution_So(t1):
  R_ijav = np.einsum('diva,jd->ijav',twoelecint_mo[occ:nao,:occ,occ-o_act:occ,occ:nao],t1)
  R_ijav += np.einsum('djav,id->ijav',twoelecint_mo[occ:nao,:occ,occ:nao,occ-o_act:occ],t1)
  R_ijav += -np.einsum('ijkv,ka->ijav',twoelecint_mo[:occ,:occ,:occ,occ-o_act:occ],t1)

  return R_ijav
  R_ijav = None
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def coupling_terms_So(So,t2):
  II_ov = np.zeros((v_act,occ)) 
  II_ov[:,occ-o_act:occ] += -2*0.25*np.einsum('dulk,lkdx->ux',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],So) + 0.25*np.einsum('dulk,kldx->ux',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],So) 
  
  R_iuab = -np.einsum('ux,xiba->iuab',II_ov,t2) 

  return R_iuab, II_ov
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def coupling_terms_Sv(Sv,t2):
  II_vo = np.zeros((virt,o_act))
  II_vo[:v_act,:] += 2*0.25*np.einsum('cblv,lwcb->wv',twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],Sv) - 0.25*np.einsum('bclv,lwcb->wv',twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],Sv)
 
  R_ijav = np.einsum('wv,jiwa->ijav',II_vo,t2)

  return R_ijav, II_vo
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def w2_int_1(So,Sv,t2):
  II_ovoo = np.zeros((occ,virt,o_act,occ))
  II_vvvo2 = np.zeros((virt,virt,virt,o_act))
  II_ovoo2 = np.zeros((occ,virt,occ,o_act))

  II_ovoo[:,:,:,occ-o_act:occ] += -np.einsum('cdvk,jkcw->jdvw',twoelecint_mo[occ:nao,occ:nao,occ-o_act:occ,:occ],So)
  II_vvvo2[:,:v_act,:,:] += -np.einsum('dckv,kxac->dxav',twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],Sv)
  II_ovoo2[:,:v_act,:,:] += np.einsum('dckv,ixdc->ixkv',twoelecint_mo[occ:nao,occ:nao,:occ,occ-o_act:occ],Sv)

  R_ijav = 2.0*np.einsum('jdvw,wida->ijav',II_ovoo,t2)
  R_ijav += -np.einsum('jdvw,wiad->ijav',II_ovoo,t2) #diagonal terms
  R_ijav += np.einsum('dxav,ijdx->ijav',II_vvvo2,t2) #off-diagonal terms
  R_ijav += -np.einsum('ixkv,kjax->ijav',II_ovoo2,t2)
  R_ijav += -np.einsum('jxkv,kixa->ijav',II_ovoo2,t2)

  return R_ijav
  gc.collect()

##-------------------------------------------------------------------------------------------------------------------------##
##-------------------------------------------------------------------------------------------------------------------------##

def w2_int_2(So,Sv,t2):
  II_vvvo = np.zeros((v_act,virt,virt,occ))
  II_ovoo3 = np.zeros((occ,v_act,occ,occ))
  II_vvvo3 = np.zeros((virt,v_act,virt,occ))
 
  II_vvvo[:,:v_act,:,:] += -np.einsum('uckl,kxbc->uxbl',twoelecint_mo[occ:occ+v_act,occ:nao,:occ,:occ],Sv) 
  II_ovoo3[:,:,:,occ-o_act:occ] += -np.einsum('dulk,ikdw->iulw',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],So)
  II_vvvo3[:,:,:,occ-o_act:occ] += -np.einsum('dulk,lkaw->duaw',twoelecint_mo[occ:nao,occ:occ+v_act,:occ,:occ],So)

  R_iuab = 2.0*np.einsum('uxbl,ilax->iuab',II_vvvo,t2)
  R_iuab += -np.einsum('uxbl,ilxa->iuab',II_vvvo,t2)
  R_iuab += -np.einsum('iulw,lwab->iuab',II_ovoo3,t2)
  R_iuab += -np.einsum('duaw,iwdb->iuab',II_vvvo3,t2) 
  R_iuab += -np.einsum('dubw,iwad->iuab',II_vvvo3,t2) 

  return R_iuab
  gc.collect()

                          ##-----------------------------------------------------------------------------------------------------------------------##
                                                                                    #THE END#
                          ##-----------------------------------------------------------------------------------------------------------------------##





