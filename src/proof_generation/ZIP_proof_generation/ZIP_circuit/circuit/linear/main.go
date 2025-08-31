package main

import (
	"bytes"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test/unsafekzg"
	"github.com/rs/zerolog"

	"gnark-float/float"
)

type FloatCircuit struct {
	A_ADD [NUM_INSTANCES_ADD]frontend.Variable `gnark:",secret"`
	B_ADD [NUM_INSTANCES_ADD]frontend.Variable `gnark:",secret"`
	C_ADD [NUM_INSTANCES_ADD]frontend.Variable `gnark:",public"`

	A_MULT [NUM_INSTANCES_MULT]frontend.Variable `gnark:",secret"`
	B_MULT [NUM_INSTANCES_MULT]frontend.Variable `gnark:",secret"`
	C_MULT [NUM_INSTANCES_MULT]frontend.Variable `gnark:",public"`

	E    uint
	M    uint
	Size uint
}

func (c *FloatCircuit) Define(api frontend.API) error {
	ctx := float.NewContext(api, c.Size, c.E, c.M)

	for instance := 0; instance < NUM_INSTANCES_ADD; instance++ {

		aFloat := ctx.NewFloat(c.A_ADD[instance])
		bFloat := ctx.NewFloat(c.B_ADD[instance])
		product := ctx.Add(aFloat, bFloat)

		expected := ctx.NewFloat(c.C_ADD[instance])
		ctx.AssertIsEqual(product, expected)

	}

	for instance2 := 0; instance2 < NUM_INSTANCES_MULT; instance2++ {

		aFloat := ctx.NewFloat(c.A_MULT[instance2])
		bFloat := ctx.NewFloat(c.B_MULT[instance2])
		product := ctx.Mul(aFloat, bFloat)

		expected := ctx.NewFloat(c.C_MULT[instance2])
		ctx.AssertIsEqual(product, expected)
	}

	return nil
}

func main() {
	zerolog.SetGlobalLevel(zerolog.Disabled)

	A_ADD_1 := make([]uint64, NUM_INSTANCES_ADD)
	B_ADD_1 := make([]uint64, NUM_INSTANCES_ADD)
	C_ADD_1 := make([]uint64, NUM_INSTANCES_ADD)

	for i := 0; i < NUM_INSTANCES_ADD; i++ {
		A_ADD_1[i] = 0xB68FFFF8000000FF //
		B_ADD_1[i] = 0x3F9080000007FFFF //
		C_ADD_1[i] = 0x3F9080000007FFFF //
	}

	A_MULT_1 := make([]uint64, NUM_INSTANCES_MULT)
	B_MULT_1 := make([]uint64, NUM_INSTANCES_MULT)
	C_MULT_1 := make([]uint64, NUM_INSTANCES_MULT)

	for i := 0; i < NUM_INSTANCES_MULT; i++ {
		A_MULT_1[i] = 0xB68FFFF8000000FF //
		B_MULT_1[i] = 0x3F9080000007FFFF //
		C_MULT_1[i] = 0xB6307FFBE0080080 //
	}

	// Convert the slices to fixed-size arrays.
	var A_ADD_1_a [NUM_INSTANCES_ADD]uint64
	var B_ADD_1_b [NUM_INSTANCES_ADD]uint64
	var C_ADD_1_c [NUM_INSTANCES_ADD]uint64
	copy(A_ADD_1_a[:], A_ADD_1)
	copy(B_ADD_1_b[:], B_ADD_1)
	copy(C_ADD_1_c[:], C_ADD_1)

	var A_ADD_1_a_s [NUM_INSTANCES_ADD]frontend.Variable
	var B_ADD_1_b_s [NUM_INSTANCES_ADD]frontend.Variable
	var C_ADD_1_c_s [NUM_INSTANCES_ADD]frontend.Variable

	for i, v := range A_ADD_1_a {
		A_ADD_1_a_s[i] = frontend.Variable(v)
	}
	for i, v := range B_ADD_1_b {
		B_ADD_1_b_s[i] = frontend.Variable(v)
	}
	for i, v := range C_ADD_1_c {
		C_ADD_1_c_s[i] = frontend.Variable(v)
	}

	var A_MULT_1_a [NUM_INSTANCES_MULT]uint64
	var B_MULT_1_b [NUM_INSTANCES_MULT]uint64
	var C_MULT_1_c [NUM_INSTANCES_MULT]uint64
	copy(A_MULT_1_a[:], A_MULT_1)
	copy(B_MULT_1_b[:], B_MULT_1)
	copy(C_MULT_1_c[:], C_MULT_1)

	var A_MULT_1_a_s [NUM_INSTANCES_MULT]frontend.Variable
	var B_MULT_1_b_s [NUM_INSTANCES_MULT]frontend.Variable
	var C_MULT_1_c_s [NUM_INSTANCES_MULT]frontend.Variable

	for i, v := range A_MULT_1_a {
		A_MULT_1_a_s[i] = frontend.Variable(v)
	}
	for i, v := range B_MULT_1_b {
		B_MULT_1_b_s[i] = frontend.Variable(v)
	}
	for i, v := range C_MULT_1_c {
		C_MULT_1_c_s[i] = frontend.Variable(v)
	}

	circuit := &FloatCircuit{ //F64
		E:    E_VALUE,
		M:    M_VALUE,
		Size: SIZE_VALUE,
	}

	r1csCircuit, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		log.Fatalf("compile error: %v", err)
	}
	totalConstraints := r1csCircuit.GetNbConstraints()
	fmt.Printf("Total number of R1CS constraints in the circuit: %d\n", totalConstraints)

	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circuit)
	if err != nil {
		log.Fatalf("compile error: %v", err)
	}

	witnessFull := &FloatCircuit{
		E:      E_VALUE,
		M:      M_VALUE,
		Size:   SIZE_VALUE,
		A_ADD:  A_ADD_1_a_s,
		B_ADD:  B_ADD_1_b_s,
		C_ADD:  C_ADD_1_c_s,
		A_MULT: A_MULT_1_a_s,
		B_MULT: B_MULT_1_b_s,
		C_MULT: C_MULT_1_c_s,
	}
	witnessPub := &FloatCircuit{
		E:      E_VALUE,
		M:      M_VALUE,
		Size:   SIZE_VALUE,
		C_ADD:  C_ADD_1_c_s,
		C_MULT: C_MULT_1_c_s,
	}

	fw, _ := frontend.NewWitness(witnessFull, ecc.BN254.ScalarField())
	pw, _ := frontend.NewWitness(witnessPub, ecc.BN254.ScalarField(), frontend.PublicOnly())

	srs, srsLagrange, err := unsafekzg.NewSRS(ccs)
	if err != nil {
		log.Fatal(err)
	}

	pk, vk, err := plonk.Setup(ccs, srs, srsLagrange)
	if err != nil {
		log.Fatalf("setup error: %v", err)
	}

	// Prove
	tProve := time.Now()
	proof, err := plonk.Prove(ccs, pk, fw)
	if err != nil {
		log.Fatalf("prove error: %v", err)
	}
	proveSecs := time.Since(tProve).Seconds()
	//fmt.Printf("Proving time: %.3f sec\n", proveSecs)

	var buf bytes.Buffer
	_, err = proof.WriteTo(&buf)
	if err != nil {
		log.Fatalf("failed to write proof: %v", err)
	}
	//fmt.Printf("Proof size: %d bytes\n", buf.Len())

	// Verify
	tVerify := time.Now()
	if err := plonk.Verify(proof, vk, pw); err != nil {
		log.Fatalf("verify error: %v", err)
	}
	verifySecs := time.Since(tVerify).Seconds()
	//fmt.Printf("Verification time: %.3f sec\n", verifySecs)

	//fmt.Println("Proof verified successfully using PLONK with float library!")

	f, err := os.OpenFile("proof_times.txt",
		os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
	if err != nil {
		log.Fatalf("open output file: %v", err)
	}
	defer f.Close()

	if _, err := fmt.Fprintf(f, "%.3f, %.3f\n", proveSecs, verifySecs); err != nil {
		log.Fatalf("write times: %v", err)
	}
}
