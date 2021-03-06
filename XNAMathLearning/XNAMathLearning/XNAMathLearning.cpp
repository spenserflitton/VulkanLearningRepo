#include "stdafx.h"
#include<Windows.h>
#include<DirectXMath.h>
#include<DirectXPackedVector.h>
#include<iostream>

using namespace DirectX;
using namespace DirectX::PackedVector;

std::ostream& operator<<(std::ostream& os, FXMVECTOR v)
{
	XMFLOAT3 dest;
	XMStoreFloat3(&dest, v);
	os << "(" << dest.x << "," << dest.y << "," << dest.z << ")";
	return os;
}

int main()
{
	std::cout.setf(std::ios_base::boolalpha);
	if (!XMVerifyCPUSupport())
	{
		std::cout << "xna math not supported" << std::endl;
		return 0;
	}

	XMVECTOR p = XMVectorZero();
	XMVECTOR q = XMVectorSplatOne();
	XMVECTOR u = XMVectorSet(1.0f, 2.0f, 3.0f, 0.0f); //The fourth parameter is meant for 4th dimension.
	XMVECTOR v = XMVectorReplicate(-2.0f);
	XMVECTOR w = XMVectorSplatZ(u);

	std::cout << "p = " << p << std::endl;
	std::cout << "q = " << q << std::endl;
	std::cout << "u = " << u << std::endl;
	std::cout << "v = " << v << std::endl;
	std::cout << "w = " << w << std::endl;

	XMVECTOR n = XMVectorSet(1.0f, 0.0f, 0.0f, 0.0f);
	XMVECTOR r = XMVectorSet(1.0f, 2.0f, 3.0f, 0.0f);
	XMVECTOR s = XMVectorSet(-2.0f, 1.0f, -3.0f, 0.0f);
	XMVECTOR t = XMVectorSet(0.707f, 0.707f, 0.0f, 0.0f);

	XMVECTOR a = r + s;
	XMVECTOR b = r - s;
	XMVECTOR c = 10.0f * r;

	std::cout << "n = " << n << std::endl;
	std::cout << "r = " << r << std::endl;
	std::cout << "s = " << s << std::endl;
	std::cout << "t = " << t << std::endl;
	std::cout << "a = " << a << std::endl;
	std::cout << "b = " << b << std::endl;
	std::cout << "c = " << c << std::endl;

	XMVECTOR rLength = XMVector3Length(r);
	XMVECTOR rNormalized = XMVector3Normalize(r);
	XMVECTOR rsDot = XMVector3Dot(r, s);
	XMVECTOR rsCross = XMVector3Cross(r, s);

	//Find proj_n(t) and perp_n(t)[orthogonalize t onto n]
	XMVECTOR projT;
	XMVECTOR perpT;
	//Using a reference normal vector, splits a 3D vector into components that are parallel and perpendicular to the normal.
	XMVector3ComponentsFromNormal(&projT, &perpT, t, n);

	//Does projT + prepT == t?
	bool equal = XMVector3Equal(projT + perpT, t) != 0;
	bool notEqual = XMVector3NotEqual(projT + perpT, t) != 0;

	XMVECTOR angleVec = XMVector3AngleBetweenVectors(projT, perpT);
	float angleRadians = XMVectorGetX(angleVec);
	float angleDegrees = XMConvertToDegrees(angleRadians);

	std::cout << "rNormalized = " << rNormalized << std::endl;
	std::cout << "rsCross = " << rsCross << std::endl;
	std::cout << "rLength = " << rLength << std::endl;
	std::cout << "rsDot = " << rsDot << std::endl;
	std::cout << "proj_n(t) = " << projT << std::endl;
	std::cout << "perp_n(t) = " << perpT << std::endl;
	std::cout << "projT + perpT == t?" << equal << std::endl;
	std::cout << "projT + perpT != t?" << notEqual << std::endl;
	std::cout << "angle in Degrees = " << angleDegrees << std::endl;
	std::cout << "angle in Radians = " << angleRadians << std::endl;

	char enterKey;
	std::cin >> enterKey;

    return 0;
}

