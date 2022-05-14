// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __PASTA_T_HPP__
#define __PASTA_T_HPP__

extern "C" {
# include "consts.c"
void pasta_mul(vec256 out, const vec256 a, const vec256 b,
                           const vec256 p, limb_t n0);
void pasta_sqr(vec256 out, const vec256 a, const vec256 p, limb_t n0);
void pasta_from(vec256 out, const vec256 a, const vec256 p, limb_t n0);
void pasta_to_scalar(pow256 ret, const vec256 a, const vec256 p, limb_t n0);
}

# include <cstdint>

template<const vec256 MOD, const limb_t M0,
         const vec256 RR, const vec256 ONE> class pasta_t {
private:
    vec256 val;

public:
    inline operator const limb_t*() const           { return val;    }
    inline operator limb_t*()                       { return val;    }
private:
    inline limb_t& operator[](size_t i)             { return val[i]; }
    inline const limb_t& operator[](size_t i) const { return val[i]; }

public:
    inline pasta_t() {}
    inline pasta_t(const uint64_t *p)
    {
        for (size_t i = 0; i < sizeof(val)/sizeof(val[0]); i++)
            val[i] = p[i];
    }

    inline void to_scalar(pow256& scalar) const
    {   pasta_to_scalar(scalar, val, MOD, M0);   }

    static inline const pasta_t& one()
    {   return *reinterpret_cast<const pasta_t*>(ONE);   }

    inline void to()
    {   pasta_mul(val, RR, val, MOD, M0);   }
    inline void from()
    {   pasta_from(val, val, MOD, M0);   }

    inline void store(uint64_t *p) const
    {
        for (size_t i = 0; i < sizeof(val)/sizeof(val[0]); i++)
            p[i] = val[i];
    }

    inline pasta_t& operator+=(const pasta_t& b)
    {
        pasta_add(val, val, b, MOD);
        return *this;
    }
    friend inline pasta_t operator+(const pasta_t& a, const pasta_t& b)
    {
        pasta_t ret;
        pasta_add(ret, a, b, MOD);
        return ret;
    }

#if 1
    inline pasta_t& operator<<=(unsigned l)
    {
        pasta_lshift(val, val, l, MOD);
        return *this;
    }
    friend inline pasta_t operator<<(const pasta_t& a, unsigned l)
    {
        pasta_t ret;
        pasta_lshift(ret, a, l, MOD);
        return ret;
    }

    inline pasta_t& operator>>=(unsigned r)
    {
        pasta_rshift(val, val, r, MOD);
        return *this;
    }
    friend inline pasta_t operator>>(pasta_t a, unsigned r)
    {
        pasta_t ret;
        pasta_rshift(ret, a, r, MOD);
        return ret;
    }
#endif

    inline pasta_t& operator-=(const pasta_t& b)
    {
        pasta_sub(val, val, b, MOD);
        return *this;
    }
    friend inline pasta_t operator-(const pasta_t& a, const pasta_t& b)
    {
        pasta_t ret;
        pasta_sub(ret, a, b, MOD);
        return ret;
    }

    inline pasta_t& cneg(bool flag)
    {
        pasta_cneg(val, val, flag, MOD);
        return *this;
    }
    friend inline pasta_t cneg(const pasta_t& a, bool flag)
    {
        pasta_t ret;
        pasta_cneg(ret, a, flag, MOD);
        return ret;
    }
    friend inline pasta_t operator-(const pasta_t& a)
    {
        pasta_t ret;
        pasta_cneg(ret, a, true, MOD);
        return ret;
    }

    inline pasta_t& operator*=(const pasta_t& a)
    {
        if (this == &a) pasta_sqr(val, val, MOD, M0);
        else            pasta_mul(val, val, a, MOD, M0);
        return *this;
    }
    friend inline pasta_t operator*(const pasta_t& a, const pasta_t& b)
    {
        pasta_t ret;
        if (&a == &b)   pasta_sqr(ret, a, MOD, M0);
        else            pasta_mul(ret, a, b, MOD, M0);
        return ret;
    }

    // simplified exponentiation, but mind the ^ operator's precedence!
    friend inline pasta_t operator^(const pasta_t& a, unsigned p)
    {
        if (p < 2) {
            abort();
        } else if (p == 2) {
            pasta_t ret;
            pasta_sqr(ret, a, MOD, M0);
            return ret;
        } else {
            pasta_t ret;
            pasta_sqr(ret, a, MOD, M0);
            for (p -= 2; p--;)
                pasta_mul(ret, ret, a, MOD, M0);
            return ret;
        }
    }
    inline pasta_t& operator^=(unsigned p)
    {
        if (p < 2) {
            abort();
        } else if (p == 2) {
            pasta_sqr(val, val, MOD, M0);
            return *this;
        }
        return *this = *this^p;
    }
    inline pasta_t operator()(unsigned p)
    {   return *this^p;   }
    friend inline pasta_t sqr(const pasta_t& a)
    {   return a^2;   }

    inline bool is_zero() const
    {   return vec_is_zero(val, sizeof(val));   }

    inline void zero()
    {
        for (size_t i=0; i < sizeof(val)/sizeof(val[0]); i++)
            val[i] = 0;
    }
};

typedef pasta_t<Vesta_P, 0x8c46eb20ffffffff, Vesta_RR, Vesta_one> vesta_t;
typedef pasta_t<Pallas_P, 0x992d30ecffffffff, Pallas_RR, Pallas_one> pallas_t;

#endif
