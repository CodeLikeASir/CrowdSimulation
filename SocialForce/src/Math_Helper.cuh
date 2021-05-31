#pragma once
#include <cstdlib>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>
#include <cuda.h>
#include <iostream>
#include <cuda_runtime.h>

__device__ inline float2 operator+(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x + rhs.x, lhs.y + rhs.y);
}

__device__ inline float2 operator-(const float2& lhs, const float2& rhs)
{
	return make_float2(lhs.x - rhs.x, lhs.y - rhs.y);
}

__device__ inline float dot(const float2& lhs, const float2& rhs)
{
	return lhs.x * rhs.x + lhs.y * rhs.y;
}

__device__ inline float2 operator*(const float lhs, const float2& rhs)
{
	return make_float2(lhs * rhs.x, lhs * rhs.y);
}

__device__ inline float2 operator*(const float2& lhs, const float rhs)
{
	return rhs * lhs;
}

__device__ inline float2 operator/(const float2& lhs, const float rhs)
{
	return make_float2(lhs.x / rhs, lhs.y / rhs);
}

__device__ inline float magnitude(const float2& val)
{
	return sqrtf(val.x * val.x + val.y * val.y);
}

__device__ inline float2 normalize(const float2 val)
{
	const float mag = sqrtf(val.x * val.x + val.y * val.y);
	return make_float2(val.x / mag, val.y / mag);
}