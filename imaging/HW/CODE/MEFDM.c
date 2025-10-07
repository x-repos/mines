/* 2D elastic time-domain FD modeling */
#include <rsf.h>
#ifdef _OPENMP
#include <omp.h>
#include "omputil.h"
#endif
#include "fdutil.h"

#define NOP 4 /* derivative operator half-size */

/* derivative coefficient (8th order) */
#define C1 +1.230862
#define C2 -0.1034123
#define C3 +0.02011671
#define C4 -0.003245760

/*  forward FD derivative stencils */
#define Fx(a,ix,iz,s) (C4*(a[ix+4][iz  ] - a[ix-3][iz  ]) +	\
								       C3*(a[ix+3][iz  ] - a[ix-2][iz  ]) +	\
								       C2*(a[ix+2][iz  ] - a[ix-1][iz  ]) +	\
                       C1*(a[ix+1][iz  ] - a[ix  ][iz  ])  )*s
#define Fz(a,ix,iz,s) (C4*(a[ix  ][iz+4] - a[ix  ][iz-3]) +	\
								       C3*(a[ix  ][iz+3] - a[ix  ][iz-2]) +	\
								       C2*(a[ix  ][iz+2] - a[ix  ][iz-1]) +	\
                       C1*(a[ix  ][iz+1] - a[ix  ][iz  ])  )*s

/* backward FD derivative stencils */
#define Bx(a,ix,iz,s) (C4*(a[ix+3][iz  ] - a[ix-4][iz  ]) +	\
								       C3*(a[ix+2][iz  ] - a[ix-3][iz  ]) +	\
								       C2*(a[ix+1][iz  ] - a[ix-2][iz  ]) +	\
                       C1*(a[ix  ][iz  ] - a[ix-1][iz  ])  )*s
#define Bz(a,ix,iz,s) (C4*(a[ix  ][iz+3] - a[ix  ][iz-4]) +	\
								       C3*(a[ix  ][iz+2] - a[ix  ][iz-3]) +	\
								       C2*(a[ix  ][iz+1] - a[ix  ][iz-2]) +	\
                       C1*(a[ix  ][iz  ] - a[ix  ][iz-1])  )*s


/* center FD derivative stencils */
#define Cx(a,ix,iz,s) (0.5f*(Fx(a,ix,iz,s) + Bx(a,ix,iz,s)))
#define Cz(a,ix,iz,s) (0.5f*(Fz(a,ix,iz,s) + Bz(a,ix,iz,s)))


/*------------------------------------------------------------*/

int main(int argc, char* argv[])
{
    bool verb,fsrf,snap,ssou,dabc;
    int  jsnap,ntsnap,jdata;

    /* I/O files */
    sf_file Fwav=NULL; /* wavelet   */
    sf_file Fsou=NULL; /* sources   */
    sf_file Frec=NULL; /* receivers */
    sf_file Fccc=NULL; /* stiffness */
    sf_file Fden=NULL; /* density   */
    sf_file Fdat=NULL; /* data      */
    sf_file Fwfl=NULL; /* wavefield */
    sf_file Fwpo=NULL; /* wavefield P- and S-like */

    /* cube axes */
    sf_axis at,ax,az;
    sf_axis as,ar,ac;

    int     nt,nz,nx,ns,nr,nc,nb;
    int     it,iz,ix;
    float   dt,dz,dx,idz,idx;

    /* FDM structure */
    fdm2d    fdm=NULL;
    abcone2d abcp=NULL,abcs=NULL;
    sponge   spo=NULL;

    /* I/O arrays */
    float***ww=NULL;           /* wavelet   */
    pt2d   *ss=NULL;           /* sources   */
    pt2d   *rr=NULL;           /* receivers */
    float **dd=NULL;           /* data      */

    float **tt=NULL;
    float **ro=NULL;           /* density   */

    /* orthorombic footprint - 4 coefficients */
    /* c11 c13
       .   c33
               c55 */

    float **c11=NULL;
    float **c33=NULL;
    float **c55=NULL;
    float **c13=NULL;
    float **vp=NULL,**vs=NULL;

    /*------------------------------------------------------------*/
    /* displacement: um = U @ t-1; uo = U @ t; up = U @ t+1 */
    float **umz,**uoz,**upz,**uaz,**utz;
    float **umx,**uox,**upx,**uax,**utx;

    /* stress/strain tensor */
    float **tzz,**tzx,**txx;
    float   szz,  szx,  sxx;

    /*------------------------------------------------------------*/
    /* linear interpolation weights/indices */
    lint2d cs,cr;

    /* Gaussian bell */
    int nbell;

    /* wavefield cut params */
    sf_axis   acz=NULL,acx=NULL;
    int       nqz,nqx;
    float     oqz,oqx;
    float     dqz,dqx;
    float     **uc=NULL;

    float **divu, **rotu;


    /*------------------------------------------------------------*/
    /* init RSF */
    sf_init(argc,argv);

    /*------------------------------------------------------------*/
    /* OMP parameters */
#ifdef _OPENMP
    omp_init();
#endif

    if(! sf_getbool("verb",&verb)) verb=false; /* verbosity flag */
    if(! sf_getbool("snap",&snap)) snap=false; /* wavefield snapshots flag */
    if(! sf_getbool("free",&fsrf)) fsrf=false; /* free surface flag */
    if(! sf_getbool("ssou",&ssou)) ssou=false; /* stress source */
    if(! sf_getbool("dabc",&dabc)) dabc=false; /* absorbing BC */

    /* top: */
    bool opot=false;
    if(! sf_getbool("opot",&opot)) opot=false;


    /*------------------------------------------------------------*/
    /* I/O files */
    Fwav = sf_input ("in" ); /* wavelet   */
    Fccc = sf_input ("ccc"); /* stiffness */
    Fden = sf_input ("den"); /* density   */
    Fsou = sf_input ("sou"); /* sources   */
    Frec = sf_input ("rec"); /* receivers */
    Fdat = sf_output("out"); /* data      */
    if(snap){
        char *wflname=NULL, *wponame=NULL;
        if(NULL==(wflname=sf_getstring("wfl"))) wflname="wfl";
        if(NULL==(wponame=sf_getstring("wpo"))) wponame="wpo";
        Fwfl = sf_output(wflname); /* wavefield */
        Fwpo = sf_output(wponame); /* wavefield P- and S-like */
    }

    /*------------------------------------------------------------*/
    /* axes */
    at = sf_iaxa(Fwav,3); sf_setlabel(at,"t"); if(verb) sf_raxa(at); /* time */
    az = sf_iaxa(Fccc,1); sf_setlabel(az,"z"); if(verb) sf_raxa(az); /* depth */
    ax = sf_iaxa(Fccc,2); sf_setlabel(ax,"x"); if(verb) sf_raxa(ax); /* space */

    as = sf_iaxa(Fsou,2); sf_setlabel(as,"s"); if(verb) sf_raxa(as); /* sources */
    ar = sf_iaxa(Frec,2); sf_setlabel(ar,"r"); if(verb) sf_raxa(ar); /* receivers */

    nt = sf_n(at); dt = sf_d(at);
    nz = sf_n(az); dz = sf_d(az);
    nx = sf_n(ax); dx = sf_d(ax);

    ns = sf_n(as);
    nr = sf_n(ar);
    /*------------------------------------------------------------*/

    /*------------------------------------------------------------*/
    /* other execution parameters */
    if(! sf_getint("nbell",&nbell)) nbell=5;  /* bell size */
    if(! sf_getint("jdata",&jdata)) jdata=1;
    if(snap) {  /* save wavefield every *jsnap* time steps */
			if(! sf_getint("jsnap",&jsnap)) jsnap=nt;
    }
    /*------------------------------------------------------------*/

    /* 2D vector components */
    nc=2;
    ac=sf_maxa(nc,0,1);

    /*------------------------------------------------------------*/
    /* setup output data header */
    sf_oaxa(Fdat,ar,1);
    sf_oaxa(Fdat,ac,2);

    sf_setn(at,nt/jdata);
    sf_setd(at,dt*jdata);
    sf_oaxa(Fdat,at,3);

    /* setup output wavefield header */
    if(snap) {
	if(!sf_getint  ("nqz",&nqz)) nqz=sf_n(az);
	if(!sf_getint  ("nqx",&nqx)) nqx=sf_n(ax);

	if(!sf_getfloat("oqz",&oqz)) oqz=sf_o(az);
	if(!sf_getfloat("oqx",&oqx)) oqx=sf_o(ax);

	dqz=sf_d(az);
	dqx=sf_d(ax);

	acz = sf_maxa(nqz,oqz,dqz);
	acx = sf_maxa(nqx,oqx,dqx);
	uc=sf_floatalloc2(sf_n(acz),sf_n(acx));

	ntsnap=0;
	for(it=0; it<nt; it++) {
	    if(it%jsnap==0) ntsnap++;
	}
	sf_setn(at,  ntsnap);
	sf_setd(at,dt*jsnap);

	sf_oaxa(Fwfl,acz,1);
	sf_oaxa(Fwfl,acx,2);
	sf_oaxa(Fwfl,ac, 3);
	sf_oaxa(Fwfl,at, 4);

	sf_oaxa(Fwpo,acz,1);
	sf_oaxa(Fwpo,acx,2);
	sf_oaxa(Fwpo,ac, 3);
	sf_oaxa(Fwpo,at, 4);
    }

    /*------------------------------------------------------------*/
    /* expand domain for FD operators and ABC */
    if( !sf_getint("nb",&nb) || nb<NOP) nb=NOP;

    fdm=fdutil_init(verb,fsrf,az,ax,nb,1);
    fdbell_init(nbell);

    sf_setn(az,fdm->nzpad); sf_seto(az,fdm->ozpad);
    sf_setn(ax,fdm->nxpad); sf_seto(ax,fdm->oxpad);
    /*------------------------------------------------------------*/

    /*------------------------------------------------------------*/
    /* source array */
    ww=sf_floatalloc3(ns,nc,nt);
    sf_floatread(ww[0][0],nt*nc*ns,Fwav);

    /* data array */
    dd=sf_floatalloc2(nr,nc);

    /*------------------------------------------------------------*/
    /* setup source/receiver coordinates */
    ss = (pt2d*) sf_alloc(ns,sizeof(*ss));
    rr = (pt2d*) sf_alloc(nr,sizeof(*rr));

    pt2dread1(Fsou,ss,ns,2); /* read (x,z) coordinates */
    pt2dread1(Frec,rr,nr,2); /* read (x,z) coordinates */

    cs = lint2d_make(ns,ss,fdm);
    cr = lint2d_make(nr,rr,fdm);

    /*------------------------------------------------------------*/
    /* setup FD coefficients */
    idz = 1/dz;
    idx = 1/dx;

    /*------------------------------------------------------------*/
    tt = sf_floatalloc2(nz,nx);
    ro =sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    c11=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    c33=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    c55=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    c13=sf_floatalloc2(fdm->nzpad,fdm->nxpad);

    /* input density */
    sf_floatread(tt[0],nz*nx,Fden);     expand(tt,ro ,fdm);

    /* input stiffness */
    sf_floatread(tt[0],nz*nx,Fccc );    expand(tt,c11,fdm);
    sf_floatread(tt[0],nz*nx,Fccc );    expand(tt,c33,fdm);
    sf_floatread(tt[0],nz*nx,Fccc );    expand(tt,c55,fdm);
    sf_floatread(tt[0],nz*nx,Fccc );    expand(tt,c13,fdm);

    free(*tt); free(tt);

    /*------------------------------------------------------------*/
    if(dabc) {
	/* one-way abc setup   */
	vp = sf_floatalloc2(fdm->nzpad,fdm->nxpad);
	vs = sf_floatalloc2(fdm->nzpad,fdm->nxpad);
	for    (ix=0; ix<fdm->nxpad; ix++) {
	    for(iz=0; iz<fdm->nzpad; iz++) {
		vp[ix][iz] = sqrt( c11[ix][iz]/ro[ix][iz] );
		vs[ix][iz] = sqrt( c55[ix][iz]/ro[ix][iz] );  /* Fixed: should be c55, not c13 */
	    }
	}

	abcp = abcone2d_make(NOP,dt,vp,fsrf,fdm);
	abcs = abcone2d_make(NOP,dt,vs,fsrf,fdm);

	/* sponge abc setup */
	spo = sponge_make(fdm->nb);
    }

    /*------------------------------------------------------------*/
    /* precompute 1/ro * dt^2 */
    for(ix=NOP; ix<fdm->nxpad-NOP; ix++) {
	for(iz=NOP; iz<fdm->nzpad-NOP; iz++) {
	    ro[ix][iz] = dt*dt/ro[ix][iz];
	}
    }

    /*------------------------------------------------------------*/
    /* allocate wavefield arrays */
    umz=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    uoz=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    upz=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    uaz=sf_floatalloc2(fdm->nzpad,fdm->nxpad);

    umx=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    uox=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    upx=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    uax=sf_floatalloc2(fdm->nzpad,fdm->nxpad);

    tzz=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    tzx=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    txx=sf_floatalloc2(fdm->nzpad,fdm->nxpad);

    divu=sf_floatalloc2(fdm->nzpad,fdm->nxpad);
    rotu=sf_floatalloc2(fdm->nzpad,fdm->nxpad);

    for    (ix=0; ix<fdm->nxpad; ix++) {
	    for(iz=0; iz<fdm->nzpad; iz++) {
	        umz[ix][iz]=0; umx[ix][iz]=0;
	        uoz[ix][iz]=0; uox[ix][iz]=0;
	        upz[ix][iz]=0; upx[ix][iz]=0;
	        uaz[ix][iz]=0; uax[ix][iz]=0;
	    }
    }

    /*------------------------------------------------------------*/
    /*
     *  MAIN LOOP
     */
    /*------------------------------------------------------------*/
    if(verb) fprintf(stderr,"\n");
    for (it=0; it<nt; it++) {
	if(verb) fprintf(stderr,"\b\b\b\b\b%d",it);

	/*------------------------------------------------------------*/
	/* from displacement to strain                                */
	/*------------------------------------------------------------*/
	/*
	 * exx = Fx(ux)
	 * ezz = Fz(uz)
	 * ezx = Bx(uz) + Bz(ux)
	 */
#ifdef _OPENMP
#pragma omp parallel for			\
    schedule(dynamic,fdm->ompchunk)		\
    private(iz,ix)				\
    shared(fdm,tzz,tzx,txx,uoz,uox,idz,idx)
#endif
	for    (ix=NOP; ix<fdm->nxpad-NOP; ix++) {
	    for(iz=NOP; iz<fdm->nzpad-NOP; iz++) {

		txx[ix][iz] = Fx(uox,ix,iz,idx);
		tzz[ix][iz] = Fz(uoz,ix,iz,idz);
	        tzx[ix][iz] = Bx(uoz,ix,iz,idx) + Bz(uox,ix,iz,idz);
	    }
	}

	/*------------------------------------------------------------*/
	/* from strain to stress                                      */
	/*------------------------------------------------------------*/
#ifdef _OPENMP
#pragma omp parallel for	    \
    schedule(dynamic,fdm->ompchunk)		\
    private(iz,ix,szz,szx,sxx)			\
    shared(fdm,tzz,tzx,txx,c11,c33,c55,c13)
#endif
	for    (ix=0; ix<fdm->nxpad; ix++) {
	    for(iz=0; iz<fdm->nzpad; iz++) {

		sxx = c11[ix][iz] * txx[ix][iz]
		    + c13[ix][iz] * tzz[ix][iz];

		szz = c13[ix][iz] * txx[ix][iz]
		    + c33[ix][iz] * tzz[ix][iz];

		szx = c55[ix][iz] * tzx[ix][iz];

		txx[ix][iz] = sxx;
		tzz[ix][iz] = szz;
		tzx[ix][iz] = szx;
	    }
	}

	/*------------------------------------------------------------*/
	/* free surface */
	/*------------------------------------------------------------*/
	if(fsrf) {
	    for    (ix=0; ix<fdm->nxpad; ix++) {
		for(iz=0; iz<fdm->nb; iz++) {
		    txx[ix][iz]=0;
		    tzz[ix][iz]=0;
		    tzx[ix][iz]=0;
		}
	    }
	}

	/*------------------------------------------------------------*/
	/* inject stress source                                       */
	/*------------------------------------------------------------*/
	if(ssou) {
	    lint2d_bell(tzz,ww[it][0],cs);
	    lint2d_bell(txx,ww[it][1],cs);
	}

	if(dabc){
	    sponge2d_apply(txx,spo,fdm);
	    sponge2d_apply(tzx,spo,fdm);
	    sponge2d_apply(tzz,spo,fdm);
	}

	/*------------------------------------------------------------*/
	/* from stress to acceleration                                */
	/*------------------------------------------------------------*/
	/*
	 * ax = Bx(txx) + Fz(txz)
	 * az = Fx(txz) + Bz(tzz)
	 */
#ifdef _OPENMP
#pragma omp parallel for			\
    schedule(dynamic,fdm->ompchunk)		\
    private(iz,ix)				\
    shared(fdm,tzz,tzx,txx,uaz,uax,idz,idx)
#endif
	for    (ix=NOP; ix<fdm->nxpad-NOP; ix++) {
	    for(iz=NOP; iz<fdm->nzpad-NOP; iz++) {
		    uax[ix][iz] = Bx( txx,ix,iz,idx ) + Fz( tzx,ix,iz,idz );
		    uaz[ix][iz] = Fx( tzx,ix,iz,idx ) + Bz( tzz,ix,iz,idz );
	    }
	}

	/*------------------------------------------------------------*/
	/* inject acceleration source                                 */
	/*------------------------------------------------------------*/
	if(!ssou) {
	    lint2d_bell(uaz,ww[it][0],cs);
	    lint2d_bell(uax,ww[it][1],cs);
	}

	/*------------------------------------------------------------*/
	/* step forward in time                                       */
	/*------------------------------------------------------------*/
#ifdef _OPENMP
#pragma omp parallel for				\
    schedule(dynamic,fdm->ompchunk)			\
    private(iz,ix)					\
    shared(fdm,uoz,uox,umz,umx,upz,upx,uaz,uax,ro)
#endif
	for    (ix=0; ix<fdm->nxpad; ix++) {
	    for(iz=0; iz<fdm->nzpad; iz++) {
		upz[ix][iz] = 2*uoz[ix][iz]
		    -           umz[ix][iz]
		    +           uaz[ix][iz] * ro[ix][iz];

		upx[ix][iz] = 2*uox[ix][iz]
		    -           umx[ix][iz]
		    +           uax[ix][iz] * ro[ix][iz];
	    }
	}

	/* circulate wavefield arrays */
	utz=umz; utx=umx;
	umz=uoz; umx=uox;
	uoz=upz; uox=upx;
	upz=utz; upx=utx;


	if(dabc) {
	    /* one-way ABC */
	    abcone2d_apply(uoz,umz,NOP,abcp,fdm);
	    abcone2d_apply(uox,umx,NOP,abcp,fdm);

	    abcone2d_apply(uoz,umz,NOP,abcs,fdm);
	    abcone2d_apply(uox,umx,NOP,abcs,fdm);

	    /* sponge ABC */
	    sponge2d_apply(umz,spo,fdm);
	    sponge2d_apply(uoz,spo,fdm);

	    sponge2d_apply(umx,spo,fdm);
	    sponge2d_apply(uox,spo,fdm);
	}

#pragma omp parallel for schedule(dynamic, fdm->ompchunk) private(ix,iz)
    for (ix = NOP; ix < fdm->nxpad - NOP; ++ix){
        for (iz = NOP; iz < fdm->nzpad - NOP; ++iz){
            divu[ix][iz] = Cx(uox, ix, iz, idx) + Cz(uoz, ix, iz, idz);        /* ∇·u  (P-like) */
            rotu[ix][iz] = Cz(uox, ix, iz, idz) - Cx(uoz, ix, iz, idx);        /* (∇×u)_y (S-like) */
        }
    }



	/*------------------------------------------------------------*/
	/* cut wavefield and save */
	/*------------------------------------------------------------*/
	if(snap && it%jsnap==0) {
	    cut2d(uoz,uc,fdm,acz,acx);
	    sf_floatwrite(uc[0],sf_n(acz)*sf_n(acx),Fwfl);

	    cut2d(uox,uc,fdm,acz,acx);
	    sf_floatwrite(uc[0],sf_n(acz)*sf_n(acx),Fwfl);

        /* cut divergence and rotation */
        cut2d(divu,uc,fdm,acz,acx);
        sf_floatwrite(uc[0],sf_n(acz)*sf_n(acx),Fwpo);
        cut2d(rotu,uc,fdm,acz,acx);
        sf_floatwrite(uc[0],sf_n(acz)*sf_n(acx),Fwpo);

	}

	// lint2d_extract(uoz,dd[0],cr);
	// lint2d_extract(uox,dd[1],cr);
	// if(it%jdata==0) sf_floatwrite(dd[0],nr*nc,Fdat);

    /* choose what the seismogram contains */
    if (opot) {
        /* potentials-based seismogram: dd[0]=div u (P-like), dd[1]=rot u (S-like) */
        lint2d_extract(divu,dd[0],cr);
        lint2d_extract(rotu,dd[1],cr);
    } else {
        /* displacement-based seismogram: dd[0]=u_z, dd[1]=u_x */
        lint2d_extract(uoz,dd[0],cr);
        lint2d_extract(uox,dd[1],cr);
    }
    if(it%jdata==0) sf_floatwrite(dd[0],nr*nc,Fdat);




    }
    if(verb) fprintf(stderr,"\n");

    /*------------------------------------------------------------*/
    /* deallocate arrays */

    free(**ww); free(*ww); free(ww);
    free(ss);
    free(rr);
    free(*dd);  free(dd);

    free(*ro);  free(ro);
    free(*c11); free(c11);
    free(*c33); free(c33);
    free(*c55); free(c55);
    free(*c13); free(c13);

    free(*umz); free(umz);
    free(*uoz); free(uoz);
    free(*upz); free(upz);
    free(*uaz); free(uaz);

    free(*umx); free(umx);
    free(*uox); free(uox);
    free(*upx); free(upx);
    free(*uax); free(uax);

    free(*tzz); free(tzz);
    free(*txx); free(txx);
    free(*tzx); free(tzx);

    free(*divu); free(divu);
    free(*rotu); free(rotu);

    if(snap) {
	free(*uc);  free(uc);
    }
    if(dabc){
	free(*vp); free(vp);
	free(*vs); free(vs);
    }

    /*------------------------------------------------------------*/

    exit (0);
}
