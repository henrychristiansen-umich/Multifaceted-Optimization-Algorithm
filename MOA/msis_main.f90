
! gfortran -c msise90_sub.for
! gfortran -c msis_main.f90
! gfortran -o msis90.exe msise90_sub.o msis_main.o
! ./msis90.exe

program msis90_main

  integer :: IYD = 24123
  real :: SEC = 0.0
  real :: ALT = 400.0
  real :: GLAT = 45.0
  real :: GLONG = 165.0
  real :: ut
  real :: STL
  real :: F107A = 120.0
  real :: F107 = 130.0
  real :: AP(7)
  integer :: MASS = 48
  real :: D(8)
  real :: T(2)
  integer :: year, doy

  integer :: lun_ = 33
  integer :: ierror
  character (len = 100) :: filein, fileout
  
  data ap/1,2,3,4,5,6,7/

  filein = 'msis_drives.txt'
  fileout = 'msis_out.txt'
  
  open(lun_, file = trim(filein), status="old", iostat=ierror)

  if (ierror .ne. 0) then
     write(*,*) 'could not find file : ', trim(filein)
     stop
  endif

  read(lun_, *, iostat=ierror) year, doy, hour, minute, second, &
       glong, glat, alt, F107, F107A, &
       (ap(j), j=1,7)
  close(lun_)
  
  ut = sec / 3600.0
  STL = mod(ut + glon/15.0, 24.0)

  call GTD6(IYD,SEC,ALT,GLAT,GLONG,STL,F107A,F107,AP,MASS,D,T)

  open(lun_, file = trim(fileout), action="write", iostat=ierror)
  if (ierror .ne. 0) then
     write(*,*) 'could not write file : ', trim(fileout)
     stop
  endif
  
  write(lun_,*) D, T
  close(lun_)
  
end program msis90_main
