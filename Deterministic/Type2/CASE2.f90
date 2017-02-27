!----------------------------------------------------------------------
!   Case2 model subroutines
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----


      IMPLICIT NONE
      INTEGER NDIM, IJAC, ICP(*)
      DOUBLE PRECISION U(NDIM), PAR(*), F(NDIM), DFDU(*), DFDP(*)
      DOUBLE PRECISION X,Y,Z, s1, s2, alpha, p1, p2, q1, q2, c1, c2, beta, gmma

	s1 = PAR(1)
	s2 = PAR(2)
	alpha = PAR(3)
	p1 = PAR(4)
	p2 = PAR(5)
	q1 = PAR(6)
	q2 = PAR(7)
	c1 = PAR(8)
	c2 = PAR(9)
	beta = PAR(10)
	gmma = PAR(11)

	X = U(1)
	Y = U(2)
	Z = U(3)

	F(1) = p1*Y+p2*(c1+c2*Z-(X**2+Y**2))*X!+gmma
	F(2) = q1*X+q2*(c1+c2*Z-(X**2+Y**2))*Y
	!F(1) = p1*Y+p2*(beta-(X**2+Y**2))*X!+gmma
	!F(2) = q1*X+q2*(beta-(X**2+Y**2))*Y
	F(3) = s1*Z**3+s2*Z+alpha

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
       PAR(1)=-1.	!s1
       PAR(2)=1.	!s2
       PAR(3)=0.	!alpha
       PAR(4)=1.	!p1
       PAR(5)=1.	!p2
       PAR(6)=-1.	!q1
       PAR(7)=1.	!q2
       PAR(8)=-0.1	!c1
       PAR(9)=0.12	!c2
       PAR(10)=0.	!beta
       PAR(11)=0	!gmma


!      Set the variables equilibria
       U(1)=0.
       U(2)=0.
       U(3)=0.

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND 
      END SUBROUTINE ICND

      SUBROUTINE FOPT 
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
