!----------------------------------------------------------------------
!   Case2 model subroutines
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----


      IMPLICIT NONE
      INTEGER NDIM, IJAC, ICP(*)
      DOUBLE PRECISION U(NDIM), PAR(*), F(NDIM), DFDU(*), DFDP(*)
      DOUBLE PRECISION X,Y,q,v, p1, p2, q1, q2, alpha, m1, m2, n1, n2, beta

	p1 = PAR(1)
	p2 = PAR(2)
	q1 = PAR(3)
	q2 = PAR(4)
	alpha = PAR(5)
	m1 = PAR(6)
	m2 = PAR(7)
	n1 = PAR(8)
	n2 = PAR(9)
	!beta = PAR(10)

	X = U(1)
	Y = U(2)
	q = U(3)
	v = U(4)

	beta = -0.05+2.*X
	F(1) = p1*Y+p2*(alpha-(X**2+Y**2))*X!+gmma
	F(2) = q1*X+q2*(alpha-(X**2+Y**2))*Y
	F(3) = m1*v+m2*(beta-(q**2+v**2))*q
	F(4) = n1*q+n2*(beta-(q**2+v**2))*v

      END SUBROUTINE FUNC
!----------------------------------------------------------------------
!----------------------------------------------------------------------

      SUBROUTINE STPNT(NDIM,U,PAR,Z)
!     ---------- ----- 

      IMPLICIT NONE
      INTEGER, INTENT(IN) :: NDIM
      DOUBLE PRECISION, INTENT(INOUT) :: U(NDIM),PAR(*)
      DOUBLE PRECISION, INTENT(IN) :: Z

!      Set the parameters
       PAR(1)=0.01	!p1
       PAR(2)=1.	!p2
       PAR(3)=-0.01	!q1
       PAR(4)=1.	!q2
       PAR(5)=-0.3	!alpha
       PAR(6)=0.1	!m1
       PAR(7)=1.	!m2
       PAR(8)=-0.1	!n1
       PAR(9)=1.	!n2
!       PAR(10)=.0	!beta


!      Set the variables equilibria
       U(1)=0.
       U(2)=0.
       U(3)=0.
       U(4)=0.

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND 
      END SUBROUTINE ICND

      SUBROUTINE FOPT 
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
