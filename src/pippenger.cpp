// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <cstring>
#include <memory>

using namespace std;

extern "C" {
#include "vect.h"
}

#include "jacobian_t.hpp"
#include "pasta_t.hpp"

typedef jacobian_t<pallas_t> point_t;
typedef point_t::affine_t affine_t;
typedef vesta_t scalar_t;

/* Works up to 25 bits. */
static limb_t get_wval_limb(const byte *d, size_t off, size_t bits)
{
    size_t i, top = (off + bits - 1)/8;
    limb_t ret, mask = (limb_t)0 - 1;

    d   += off/8;
    top -= off/8-1;

    /* this is not about constant-time-ness, but branch optimization */
    for (ret=0, i=0; i<4;) {
        ret |= (*d & mask) << (8*i);
        mask = (limb_t)0 - ((++i - top) >> (8*sizeof(top)-1));
        d += 1 & mask;
    }

    return ret >> (off%8);
}

static size_t window_size(size_t npoints)
{
    size_t wbits;

    for (wbits=0; npoints>>=1; wbits++) ;

    return wbits>12 ? wbits-3 : (wbits>4 ? wbits-2 : (wbits ? 2 : 1));
}

static void integrate_buckets(point_t& out, point_t buckets[], size_t wbits)
{
    point_t ret, acc;
    size_t n = (size_t)1 << wbits;

    /* Calculate sum of x[i-1]*i for i=1 through 1<<|wbits|. */
    vec_copy(&acc, &buckets[--n], sizeof(acc));
    vec_copy(&ret, &buckets[n], sizeof(ret));
    vec_zero(&buckets[n], sizeof(buckets[n]));
    while (n--) {
        acc.add(buckets[n]);
        ret.add(acc);
        vec_zero(&buckets[n], sizeof(buckets[n]));
    }
    out = ret;
}

static void bucket(point_t buckets[], limb_t booth_idx,
                   size_t wbits, const affine_t& p)
{
    booth_idx &= (1<<wbits) - 1;
    if (booth_idx--)
        buckets[booth_idx].add(p);
}

static void prefetch(const point_t buckets[], limb_t booth_idx, size_t wbits)
{
#if 0
    booth_idx &= (1<<wbits) - 1;
    if (booth_idx--)
        vec_prefetch(&buckets[booth_idx], sizeof(buckets[booth_idx]));
#else
    (void)buckets;
    (void)booth_idx;
    (void)wbits;
#endif
}

static void tile(point_t& ret, const affine_t points[], size_t npoints,
                 const byte* scalars, size_t nbits,
                 point_t buckets[], size_t bit0, size_t wbits, size_t cbits)
{
    limb_t wmask, wval, wnxt;
    size_t i, nbytes;

    nbytes = (nbits + 7)/8; /* convert |nbits| to bytes */
    wmask = ((limb_t)1 << wbits) - 1;
    wval = get_wval_limb(scalars, bit0, wbits) & wmask;
    scalars += nbytes;
    wnxt = get_wval_limb(scalars, bit0, wbits) & wmask;
    npoints--;  /* account for prefetch */

    bucket(buckets, wval, cbits, points[0]);
    for (i = 1; i < npoints; i++) {
        wval = wnxt;
        scalars += nbytes;
        wnxt = get_wval_limb(scalars, bit0, wbits) & wmask;
        prefetch(buckets, wnxt, cbits);
        bucket(buckets, wval, cbits, points[i]);
    }
    bucket(buckets, wnxt, cbits, points[i]);
    integrate_buckets(ret, buckets, cbits);
}

extern "C"
void mult_pippenger(point_t& ret, const affine_t points[], size_t npoints,
                    const scalar_t _scalars[], bool mont)
{
    const size_t nbits = 255;
    size_t wbits, cbits, bit0 = nbits;
    size_t window = window_size(npoints);

    vector<point_t> buckets(1 << window);
    memset(&buckets[0], 0, sizeof(buckets[0]) * buckets.size());

    // below is little-endian dependency, should it be removed?
    const pow256* scalars = reinterpret_cast<decltype(scalars)>(_scalars);
    unique_ptr<pow256[]> store;
    if (mont) {
        store = decltype(store)(new pow256[npoints]);
        for (size_t i = 0; i < npoints; i++)
            _scalars[i].to_scalar(store[i]);
        scalars = &store[0];
    }

    point_t p;
    ret.inf();

    /* top excess bits modulo target window size */
    wbits = nbits % window; /* yes, it may be zero */
    cbits = wbits + 1;
    while (bit0 -= wbits) {
        tile(p, points, npoints, scalars[0], 255,
                &buckets[0], bit0, wbits, cbits);
        ret.add(p);
        for (size_t i = 0; i < window; i++)
            ret.dbl();
        cbits = wbits = window;
    }
    tile(p, points, npoints, scalars[0], 255,
            &buckets[0], 0, wbits, cbits);
    ret.add(p);
}
