// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include <cstring>
#include <memory>
#include <tuple>

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

template<typename T>
static size_t num_bits(T l)
{
    const size_t T_BITS = 8*sizeof(T);
# define MSB(x) ((T)(x) >> (T_BITS-1))
    T x, mask;

    if ((T)-1 < 0) {    // handle signed T
        mask = MSB(l);
        l ^= mask;
        l += 1 & mask;
    }

    size_t bits = (((T)(~l & (l-1)) >> (T_BITS-1)) & 1) ^ 1;

    if (sizeof(T) > 4) {
        x = l >> (32 & (T_BITS-1));
        mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
        bits += 32 & mask;
        l ^= (x ^ l) & mask;
    }

    if (sizeof(T) > 2) {
        x = l >> 16;
        mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
        bits += 16 & mask;
        l ^= (x ^ l) & mask;
    }

    if (sizeof(T) > 1) {
        x = l >> 8;
        mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
        bits += 8 & mask;
        l ^= (x ^ l) & mask;
    }

    x = l >> 4;
    mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
    bits += 4 & mask;
    l ^= (x ^ l) & mask;

    x = l >> 2;
    mask = MSB(0 - x);  if ((T)-1 > 0) mask = 0 - mask;
    bits += 2 & mask;
    l ^= (x ^ l) & mask;

    bits += l >> 1;

    return bits;
# undef MSB
}

tuple<size_t, size_t, size_t>
static breakdown(size_t nbits, size_t window, size_t ncpus)
{
    size_t nx, ny, wnd;

    if (nbits > window * ncpus) {
        nx = 1;
        wnd = window - num_bits(ncpus / 4);
    } else {
        nx = 2;
        wnd = window - 2;
        while ((nbits / wnd + 1) * nx < ncpus) {
            nx += 1;
            wnd = window - num_bits(3 * nx / 2);
        }
        nx -= 1;
        wnd = window - num_bits(3 * nx / 2);
    }
    ny = nbits / wnd + 1;
    wnd = nbits / ny + 1;

    return make_tuple(nx, ny, wnd);
}

#include "thread_pool_t.hpp"
static thread_pool_t da_pool;

extern "C"
void mult_pippenger(point_t& ret, const affine_t points[], size_t npoints,
                     const scalar_t _scalars[], bool mont)
{
    const size_t nbits = 255;
    size_t window = window_size(npoints);

    // below is little-endian dependency, should it be removed?
    const pow256* scalars = reinterpret_cast<decltype(scalars)>(_scalars);
    unique_ptr<pow256[]> store = nullptr;
    if (mont) {
        store = decltype(store)(new pow256[npoints]);
        for (size_t i = 0; i < npoints; i++)
            _scalars[i].to_scalar(store[i]);
        scalars = &store[0];
    }

    size_t ncpus = da_pool.size();
    if (ncpus < 2 || npoints < 32) {
        vector<point_t> buckets(1 << window); /* zeroed */

        point_t p;
        ret.inf();

        /* top excess bits modulo target window size */
        size_t wbits = nbits % window, /* yes, it may be zero */
               cbits = wbits + 1,
               bit0 = nbits;
        while (bit0 -= wbits) {
            tile(p, points, npoints, scalars[0], nbits,
                    &buckets[0], bit0, wbits, cbits);
            ret.add(p);
            for (size_t i = 0; i < window; i++)
                ret.dbl();
            cbits = wbits = window;
        }
        tile(p, points, npoints, scalars[0], nbits,
                &buckets[0], 0, wbits, cbits);
        ret.add(p);
        return;
    }

    size_t nx, ny;
    tie(nx, ny, window) = breakdown(nbits, window, ncpus);

    struct tile_t {
        size_t x, dx, y, dy;
        point_t p;
        tile_t() {}
    };
    vector<tile_t> grid(nx * ny);

    size_t dx = npoints / nx,
           y  = window * (ny - 1);

    size_t total = 0;
    while (total < nx) {
        grid[total].x  = total * dx;
        grid[total].dx = dx;
        grid[total].y  = y;
        grid[total].dy = nbits - y;
        total++;
    }
    grid[total - 1].dx = npoints - grid[total - 1].x;

    while (y) {
        y -= window;
        for (size_t i = 0; i < nx; i++, total++) {
            grid[total].x  = grid[i].x;
            grid[total].dx = grid[i].dx;
            grid[total].y  = y;
            grid[total].dy = window;
        }
    }

    vector<atomic<size_t>> row_sync(ny); /* zeroed */
    atomic<size_t> counter(0);
    channel_t<size_t> ch;

    auto n_workers = min(ncpus, total);
    while (n_workers--) {
        da_pool.spawn([&, window, total, nbits]() {
            vector<point_t> buckets(1 << window); /* zeroed */

            for (size_t work; (work = counter++) < total;) {
                size_t x  = grid[work].x,
                       dx = grid[work].dx,
                       y  = grid[work].y,
                       dy = grid[work].dy;
                tile(grid[work].p, &points[x], dx,
                                   scalars[x], nbits, &buckets[0],
                                   y, dy, dy + (dy < window));
                if (++row_sync[y / window] == nx)
                    ch.send(y);
            }
        });
    }

    ret.inf();
    size_t row = 0;
    while (ny--) {
        auto y = ch.recv();
        row_sync[y / window] = -1U;
        while (grid[row].y == y) {
            while (row < total && grid[row].y == y)
                ret.add(grid[row++].p);
            if (y == 0)
                break;
            for (size_t i = 0; i < window; i++)
                ret.dbl();
            y -= window;
            if (row_sync[y / window] != -1U)
                break;
        }
    }
}
