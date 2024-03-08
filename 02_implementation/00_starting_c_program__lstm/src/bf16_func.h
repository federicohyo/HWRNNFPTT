#ifndef _BF16_FUNC_H_
#define _BF16_FUNC_H_

/*
    As CPU (Rocket) core does not support BF16 calculation
    to process data in BF16 (represented in C type uint16_t)
        1. convert uint16_t to FP32 
            uint16_t -> uint32_t, by shifting left 16 bits
            uint32_t -> float, by pointer type casting
        2. process the converted data in FP32
        3. convet the result back to BF16, by shifting right 16 bits
*/



uint16_t bf16_add(uint16_t a, uint16_t b)
{
    // register this result, otherwise the following pointer cast not possible
    uint32_t tmp_a = u16_to_u32(a); 
    uint32_t tmp_b = u16_to_u32(b);

    float tmp_c = u32_to_fp32(tmp_a) + u32_to_fp32(tmp_b);
    
    return fp32_to_u16(tmp_c);
}

uint16_t bf16_sub(uint16_t a, uint16_t b)
{
    uint32_t tmp_a = u16_to_u32(a);
    uint32_t tmp_b = u16_to_u32(b);

    float tmp_c = u32_to_fp32(tmp_a) - u32_to_fp32(tmp_b);
    
    return fp32_to_u16(tmp_c);
}

uint16_t bf16_mul(uint16_t a, uint16_t b)
{
    uint32_t tmp_a = u16_to_u32(a);
    uint32_t tmp_b = u16_to_u32(b);

    float tmp_c = u32_to_fp32(tmp_a) * u32_to_fp32(tmp_b);
    
    return fp32_to_u16(tmp_c);
}

uint16_t bf16_div(uint16_t a, uint16_t b)
{
    uint32_t tmp_a = u16_to_u32(a);
    uint32_t tmp_b = u16_to_u32(b);

    float tmp_c = u32_to_fp32(tmp_a) / u32_to_fp32(tmp_b);
    
    return fp32_to_u16(tmp_c);
}




uint16_t bf16_tanh(uint16_t in_dat)
{
    uint32_t tmp_in = u16_to_u32(in_dat);

    float out = tanhf( u32_to_fp32(tmp_in) );

    return fp32_to_u16(out);
}


uint16_t bf16_sigmoid(uint16_t in_dat)
{
    uint32_t tmp_in = u16_to_u32(in_dat);

    float out = 1.0f / (1 + exp(0.0f - u32_to_fp32(tmp_in)));

    return fp32_to_u16(out);
}

uint16_t bf16_greater(uint16_t a, uint16_t b)
{
    uint32_t tmp_a = u16_to_u32(a);
    uint32_t tmp_b = u16_to_u32(b);

    return u32_to_fp32(tmp_a) > u32_to_fp32(tmp_b);
}

uint16_t bf16_exp(uint16_t in_dat)
{
    uint32_t tmp_in = u16_to_u32(in_dat);

    float out = exp(u32_to_fp32(tmp_in));

    return fp32_to_u16(out);
}


#endif//_BF16_FUNC_H_