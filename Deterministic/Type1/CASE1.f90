!----------------------------------------------------------------------
!   MAOOAM model subroutines
!----------------------------------------------------------------------

      SUBROUTINE FUNC(NDIM,U,ICP,PAR,IJAC,F,DFDU,DFDP)
!     ---------- ----


      IMPLICIT NONE
      INTEGER NDIM, IJAC, ICP(*)
      DOUBLE PRECISION U(NDIM), PAR(*), F(NDIM), DFDU(*), DFDP(*)
      DOUBLE PRECISION X,Y, p1, p2, alpha, q1, q2, c1, c2, beta

	p1 = PAR(1)
	p2 = PAR(2)
	alpha = PAR(3)
	q1 = PAR(4)
	q2 = PAR(5)
	c1 = PAR(6)
	c2 = PAR(7)
	beta = PAR(8)

	X = U(1)
	Y = U(2)

	F(1) = p1*X**3.+p2*X+alpha
	F(2) = q1*Y**3.+q2*Y+c1+c2*X
	!F(2) = q1*Y**3.+q2*Y+beta

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
       PAR(1)=-0.5	!p1
       PAR(2)=0.5	!p2
       PAR(3)=0.	!alpha
       PAR(4)=-0.5	!q1
       PAR(5)=1.	!q2
       PAR(6)=0.	!c1
       PAR(7)=0.48	!c2
       PAR(8)=0.	!beta


!      Set the variables equilibria
       U(1)=0.
       U(2)=0.

      END SUBROUTINE STPNT

      SUBROUTINE BCND
      END SUBROUTINE BCND

      SUBROUTINE ICND 
      END SUBROUTINE ICND

      SUBROUTINE FOPT 
      END SUBROUTINE FOPT

      SUBROUTINE PVLS
      END SUBROUTINE PVLS
