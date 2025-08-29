package main

import (
	"fmt"
	"log"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/groth16"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/tumberger/zk-Location/float"
)

// MulCircuit defines a circuit to prove the multiplication of two floating‑point numbers.
// It uses the external float library to perform the floating‑point operations.
type MulCircuit struct {
	// Secret inputs
	A frontend.Variable `gnark:",secret"`
	B frontend.Variable `gnark:",secret"`
	// Public output: expected product of A and B
	C frontend.Variable `gnark:",public"`

	// Floating‑point parameters.
	// For F32, use E = 8 and M = 23. For F64, use E = 11 and M = 52.
	E    uint
	M    uint
	size uint // a parameter controlling the internal representation/precision (as used in the float library)
}

// Define defines the circuit.
// It creates a float context using the provided parameters, converts A and B to float representations,
// computes the multiplication, and asserts that the result equals the public output C.
func (c *MulCircuit) Define(api frontend.API) error {
	// Create a new floating‑point context using the float library.
	ctx := float.NewContext(api, c.size, c.E, c.M)

	// Convert secret inputs into floating‑point representations.
	aFloat := ctx.NewFloat(c.A)
	bFloat := ctx.NewFloat(c.B)

	// Compute the multiplication of the two floating‑point numbers.
	prod := ctx.Mul(aFloat, bFloat)

	// Expose the computed product as the public output.
	// (If necessary, convert prod from its internal format to a plain variable.)
	api.AssertIsEqual(prod, c.C)

	return nil
}

func main() {
	// Set the floating‑point parameters. Here we choose F32 (E=8, M=23) and an arbitrary size.
	circuit := &MulCircuit{
		E:    8,
		M:    23,
		size: 16,
	}

	// Compile the circuit into a R1CS using gnark’s r1cs builder.
	r1csCircuit, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		log.Fatal("Circuit compilation error:", err)
	}

	// For demonstration, we choose sample values:
	// Let A = 1.5 and B = 2.0 so that the expected product C = 3.0.
	// (Depending on the float library, these values may be automatically converted to the proper internal representation.)
	witness := &MulCircuit{
		A:    1.5,
		B:    2.0,
		C:    3.0,
		E:    8,
		M:    23,
		size: 16,
	}

	// Generate the proving and verifying keys using Groth16.
	pk, vk, err := groth16.Setup(r1csCircuit)
	if err != nil {
		log.Fatal("Setup error:", err)
	}

	// Generate a proof for the witness.
	proof, err := groth16.Prove(r1csCircuit, pk, witness)
	if err != nil {
		log.Fatal("Proving error:", err)
	}

	// Prepare the public witness. In this circuit, only the output C is public.
	publicWitness := &MulCircuit{
		C:    3.0,
		E:    8,
		M:    23,
		size: 16,
	}

	// Verify the proof.
	err = groth16.Verify(proof, vk, publicWitness)
	if err != nil {
		log.Fatal("Proof verification failed:", err)
	}

	fmt.Println("Proof verified successfully!")
}
