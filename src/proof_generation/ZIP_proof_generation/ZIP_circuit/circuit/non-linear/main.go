package main

import (
	"bytes"
	"fmt"
	"log"
	"path/filepath"

	"github.com/consensys/gnark-crypto/ecc"
	"github.com/consensys/gnark/backend/plonk"
	"github.com/consensys/gnark/frontend"
	"github.com/consensys/gnark/frontend/cs/r1cs"
	"github.com/consensys/gnark/frontend/cs/scs"
	"github.com/consensys/gnark/test/unsafekzg"
	"github.com/rs/zerolog"

	"gnark-float/float"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

type FloatCircuit struct {
	Y_prime [NUM_INSTANCES]frontend.Variable `gnark:",secret"`
	Y       [NUM_INSTANCES]frontend.Variable `gnark:",secret"`
	Delta   frontend.Variable                `gnark:",public"`

	// New fields for the lookup argument:
	A_vec [NUM_INSTANCES][PRIVATE_VECTOR_SIZE]frontend.Variable             `gnark:",secret"` // private vector a
	S_vec [NUM_INSTANCES][PRIVATE_VECTOR_SIZE][TABLE_SIZE]frontend.Variable `gnark:",secret"` // indicator vectors s0,…,s9
	C_vec [TABLE_SIZE]frontend.Variable                                     `gnark:",public"` // public table c

	A_prime_vec [NUM_INSTANCES][PRIVATE_VECTOR_PRIME_SIZE]frontend.Variable                   `gnark:",secret"` // private vector a'
	S_prime_vec [NUM_INSTANCES][PRIVATE_VECTOR_PRIME_SIZE][TABLE_PRIME_SIZE]frontend.Variable `gnark:",secret"` // indicator vectors s'0,s'1
	C_prime_vec [TABLE_PRIME_SIZE]frontend.Variable                                           `gnark:",public"` // public table c'

	// Parameters for the float context (if still used)
	E    uint
	M    uint
	Size uint
}

func (c *FloatCircuit) Define(api frontend.API) error {
	ctx := float.NewContext(api, c.Size, c.E, c.M)

	for instance := 0; instance < NUM_INSTANCES; instance++ {
		n := TABLE_SIZE
		k_hat := PRIVATE_VECTOR_SIZE
		if n%k_hat != 0 {
			return fmt.Errorf("public table size (%d) must be divisible by the number of indicator vectors (%d)", n, k_hat)
		}
		m := n / k_hat // Number of blocks. (public information)

		// Accumulate the sum for the one-hot condition.
		for j := 0; j < k_hat; j++ {
			sumS := frontend.Variable(0)
			for i := 0; i < TABLE_SIZE; i++ {
				sumS = api.Add(sumS, c.S_vec[instance][j][i])
			}
			api.AssertIsEqual(sumS, frontend.Variable(1))
		}

		// Enforce: s_j[i] * (s_j[i] - 1) = 0 (i.e. s_j[i] ∈ {0,1})
		for j := 0; j < k_hat; j++ {
			for i := 0; i < TABLE_SIZE; i++ {
				oneMinus := api.Sub(c.S_vec[instance][j][i], frontend.Variable(1))
				api.AssertIsEqual(api.Mul(c.S_vec[instance][j][i], oneMinus), frontend.Variable(0))
			}
		}

		// Enforce: s_j[(i-1)(k+1)+j] == s_{j-1}[(i-1)(k+1)+(j-1)]
		for i := 1; i <= m; i++ {
			for j := 1; j < k_hat; j++ {
				api.AssertIsEqual(c.S_vec[instance][j][(i-1)*k_hat+j], c.S_vec[instance][j-1][(i-1)*k_hat+(j-1)])
			}
		}

		// Enforce: s_j[(i-1)(k+1)+j] = (k+1)
		out := frontend.Variable(0)
		for i := 1; i <= m; i++ {
			for j := 0; j < k_hat; j++ {
				out = api.Add(out, c.S_vec[instance][j][(i-1)*k_hat+j])
			}
		}
		api.AssertIsEqual(out, frontend.Variable(k_hat))

		// Enforce: a[j] = ∑ᵢ s_j[i] * c[i]
		for j := 0; j < k_hat; j++ {
			sumVal := frontend.Variable(0)
			for i := 0; i < TABLE_SIZE; i++ {
				sumVal = api.Add(sumVal, api.Mul(c.S_vec[instance][j][i], c.C_vec[i]))
			}
			api.AssertIsEqual(c.A_vec[instance][j], sumVal)
		}

		// Enforce the first space of s_k to be 0 and last space of s_0 to be 0
		api.AssertIsEqual(c.S_vec[instance][0][TABLE_SIZE-1], frontend.Variable(0))
		api.AssertIsEqual(c.S_vec[instance][k_hat-1][0], frontend.Variable(0))

		//////////////////////////// Range Proof

		// Accumulate the sum for the one-hot condition.
		for j := 0; j < PRIVATE_VECTOR_PRIME_SIZE; j++ {
			sumS_prime := frontend.Variable(0)
			for i := 0; i < TABLE_PRIME_SIZE; i++ {
				sumS_prime = api.Add(sumS_prime, c.S_prime_vec[instance][j][i])
			}
			api.AssertIsEqual(sumS_prime, frontend.Variable(1))
		}

		// Enforce: s'_j[i] * (s'_j[i] - 1) = 0 (i.e. s'_j[i] ∈ {0,1})
		for j := 0; j < PRIVATE_VECTOR_PRIME_SIZE; j++ {
			for i := 0; i < TABLE_PRIME_SIZE; i++ {
				oneMinus_prime := api.Sub(c.S_prime_vec[instance][j][i], frontend.Variable(1))
				api.AssertIsEqual(api.Mul(c.S_prime_vec[instance][j][i], oneMinus_prime), frontend.Variable(0))
			}
		}

		// Enforce: s'_j[i+j-1] == s'_{j-1}[(i + j - 2)]
		for i := 1; i <= m; i++ {
			for j := 1; j < PRIVATE_VECTOR_PRIME_SIZE; j++ {
				api.AssertIsEqual(c.S_prime_vec[instance][j][i+j-1], c.S_prime_vec[instance][j-1][i+j-2])
			}
		}

		// Enforce the first space of s'_1 to be 0 and last space of s'_0 to be 0
		api.AssertIsEqual(c.S_prime_vec[instance][0][m], frontend.Variable(0))
		api.AssertIsEqual(c.S_prime_vec[instance][1][0], frontend.Variable(0))

		// Enforce: a'[j] = ∑ᵢ s'_j[i] * c'[i]
		for j := 0; j < PRIVATE_VECTOR_PRIME_SIZE; j++ {
			sumVal_prime := frontend.Variable(0)
			for i := 0; i < TABLE_PRIME_SIZE; i++ {
				sumVal_prime = api.Add(sumVal_prime, api.Mul(c.S_prime_vec[instance][j][i], c.C_prime_vec[i]))
			}
			api.AssertIsEqual(c.A_prime_vec[instance][j], sumVal_prime)
		}

		// Apply Range Proof

		// Enforce (y' - t_{e-1}) >= 0 and (t_e - y') >= 0
		y_primeFloat := ctx.NewFloat(c.Y_prime[instance])

		t_eMinusOne := c.A_prime_vec[instance][0]
		t_e := c.A_prime_vec[instance][1]

		t_eMinusOneFloat := ctx.NewFloat(t_eMinusOne)
		t_eFloat := ctx.NewFloat(t_e)

		zFloat1 := ctx.Sub(y_primeFloat, t_eMinusOneFloat)
		zFloat2 := ctx.Sub(t_eFloat, y_primeFloat)

		//Enforce zFloat1.Sign == 0  OR  zFloat1.Mantissa == 0
		isSignZero1 := api.IsZero(zFloat1.Sign)
		isMantissaZero1 := api.IsZero(zFloat1.Mantissa)

		// If either is true, allow it.
		api.AssertIsEqual(api.Or(isSignZero1, isMantissaZero1), 1)

		//Enforce zFloat2.Sign == 0  OR  zFloat2.Mantissa == 0
		isSignZero2 := api.IsZero(zFloat2.Sign)
		isMantissaZero2 := api.IsZero(zFloat2.Mantissa)

		// If either is true, allow it.
		api.AssertIsEqual(api.Or(isSignZero2, isMantissaZero2), 1)

		//////////////////////////// Approximation Relation

		// Enforce |y - f(y')| <= delta * |f(y')|
		yFloat := ctx.NewFloat(c.Y[instance])
		deltaFloat := ctx.NewFloat(c.Delta)

		// Evaluate the polynomial using Horner's method:
		// polyOutputFloat = f(y') = c9 * y'^9 + c8 * y'^8 + c7 * y'^7 + c6 * y'^6 + c5 * y'^5 +
		//        c4 * y'^4 + c3 * y'^3 + c2 * y'^2 + c1 * y' + c0

		k := k_hat - 1 // highest degree index, k
		acc := ctx.NewFloat(c.A_vec[instance][k])

		for i := k - 1; i >= 0; i-- {
			acc = ctx.Add(ctx.NewFloat(c.A_vec[instance][i]), ctx.Mul(y_primeFloat, acc))
		}
		polyOutputFloat := acc

		// Enforce |yFloat - polyOutputFloat|
		left := ctx.Sub(yFloat, polyOutputFloat)

		// Enforce |deltaFloat * polyOutputFloat|
		right := ctx.Mul(deltaFloat, polyOutputFloat)

		// Enforce Range
		firstSub := ctx.Add(left, right)
		secondSub := ctx.Sub(right, left)
		newZFloat := ctx.Mul(firstSub, secondSub)

		// Enforce newZFloat.Sign == 0  OR  newZFloat.Mantissa == 0
		isSignZero_newZ := api.IsZero(newZFloat.Sign)
		isMantissaZero_newZ := api.IsZero(newZFloat.Mantissa)

		// If either is true, allow it.
		api.AssertIsEqual(api.Or(isSignZero_newZ, isMantissaZero_newZ), 1)
	}
	return nil
}

func loadUint64sFromFile(path string) ([]uint64, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	txt := string(b)

	re := regexp.MustCompile(`0[xX][0-9a-fA-F]+|[0-9]+`)
	toks := re.FindAllString(txt, -1)

	out := make([]uint64, 0, len(toks))
	for _, t := range toks {
		var v uint64
		if strings.HasPrefix(t, "0x") || strings.HasPrefix(t, "0X") {
			v, err = strconv.ParseUint(t[2:], 16, 64)
		} else {
			v, err = strconv.ParseUint(t, 10, 64)
		}
		if err != nil {
			return nil, fmt.Errorf("parse %q in %s: %w", t, path, err)
		}
		out = append(out, v)
	}
	return out, nil
}

func parseUint64Token(t string) (uint64, error) {
	t = strings.TrimSpace(t)
	if strings.HasPrefix(t, "0x") || strings.HasPrefix(t, "0X") {
		return strconv.ParseUint(t[2:], 16, 64)
	}
	return strconv.ParseUint(t, 10, 64)
}

func parseIndexToken(t string) (int, error) {
	t = strings.TrimSpace(t)
	if t == "" {
		return 0, fmt.Errorf("empty index token")
	}
	var u uint64
	var err error
	if strings.HasPrefix(t, "0x") || strings.HasPrefix(t, "0X") {
		u, err = strconv.ParseUint(t[2:], 16, 64)
	} else {
		u, err = strconv.ParseUint(t, 10, 64)
	}
	if err != nil {
		return 0, err
	}
	if u > uint64(^uint(0)>>1) {
		return 0, fmt.Errorf("index too large for int: %s", t)
	}
	return int(u), nil
}

func loadPairsAndThirds(path string, need int) ([]uint64, []uint64, []int, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, nil, err
	}
	lines := strings.Split(string(b), "\n")

	Y := make([]uint64, 0, need)
	Yp := make([]uint64, 0, need)
	thirds := make([]int, 0, need)

	for _, ln := range lines {
		ln = strings.TrimSpace(ln)
		if ln == "" || strings.HasPrefix(ln, "#") {
			continue
		}
		parts := strings.Split(ln, ",")
		if len(parts) < 3 {
			return nil, nil, nil, fmt.Errorf("bad line (need 3 values Y, Y', third): %q", ln)
		}
		y, err := parseUint64Token(parts[0])
		if err != nil {
			return nil, nil, nil, fmt.Errorf("parse Y in %q: %w", ln, err)
		}
		yp, err := parseUint64Token(parts[1])
		if err != nil {
			return nil, nil, nil, fmt.Errorf("parse Y' in %q: %w", ln, err)
		}
		t, err := parseIndexToken(parts[2])
		if err != nil {
			return nil, nil, nil, fmt.Errorf("parse third in %q: %w", ln, err)
		}

		Y = append(Y, y)
		Yp = append(Yp, yp)
		thirds = append(thirds, t)

		if len(Y) == need {
			break
		}
	}
	if len(Y) < need {
		return nil, nil, nil, fmt.Errorf("file %s has only %d triples, need %d", path, len(Y), need)
	}
	return Y, Yp, thirds, nil
}

func buildIndicesFromThirds(thirds []int) ([][]int, [][]int, error) {
	if len(thirds) != NUM_INSTANCES {
		return nil, nil, fmt.Errorf("thirds length %d != NUM_INSTANCES %d", len(thirds), NUM_INSTANCES)
	}

	lookup := make([][]int, NUM_INSTANCES)
	rangeProof := make([][]int, NUM_INSTANCES)

	for i, t := range thirds {
		if t < 0 {
			return nil, nil, fmt.Errorf("third at instance %d is negative: %d", i, t)
		}

		// lookup indices: t*PV_SIZE .. t*PV_SIZE + PV_SIZE-1
		start := t * PRIVATE_VECTOR_SIZE
		end := start + PRIVATE_VECTOR_SIZE - 1
		if end >= TABLE_SIZE {
			return nil, nil, fmt.Errorf("instance %d: lookup indices [%d..%d] out of TABLE_SIZE=%d", i, start, end, TABLE_SIZE)
		}
		l := make([]int, PRIVATE_VECTOR_SIZE)
		for j := 0; j < PRIVATE_VECTOR_SIZE; j++ {
			l[j] = start + j
		}
		lookup[i] = l

		// range-proof indices: t, t+1, ..., t+(PV_PRIME_SIZE-1)
		r := make([]int, PRIVATE_VECTOR_PRIME_SIZE)
		for j := 0; j < PRIVATE_VECTOR_PRIME_SIZE; j++ {
			r[j] = t + j
		}
		if r[len(r)-1] >= TABLE_PRIME_SIZE {
			return nil, nil, fmt.Errorf("instance %d: range-proof last idx %d out of TABLE_PRIME_SIZE=%d", i, r[len(r)-1], TABLE_PRIME_SIZE)
		}
		rangeProof[i] = r
	}

	return lookup, rangeProof, nil
}

func main() {

	zerolog.SetGlobalLevel(zerolog.Disabled)

	fmt.Printf("# Activation Functions (%s): %d\n", ACTIVATION, NUM_INSTANCES)

	/*
		if !PROVING {
			zerolog.SetGlobalLevel(zerolog.ErrorLevel)
		}
	*/

	tableDir := "../../../precomputed_lookup_tables_ieee754_hex"
	intervalsPath := filepath.Join(tableDir, fmt.Sprintf("%s_intervals_ieee754.txt", ACTIVATION))
	coeffsPath := filepath.Join(tableDir, fmt.Sprintf("%s_coefficients_ieee754.txt", ACTIVATION))

	// Load tables
	publicTableC_prime, err := loadUint64sFromFile(intervalsPath)
	if err != nil {
		log.Fatalf("failed to load intervals table %q: %v", intervalsPath, err)
	}
	publicTableC, err := loadUint64sFromFile(coeffsPath)
	if err != nil {
		log.Fatalf("failed to load coefficients table %q: %v", coeffsPath, err)
	}

	// Sanity-check sizes against compile-time constants used in your circuit struct.
	if len(publicTableC) != TABLE_SIZE {
		log.Fatalf("coefficients length (%d) != TABLE_SIZE (%d). File: %s", len(publicTableC), TABLE_SIZE, coeffsPath)
	}
	if len(publicTableC_prime) != TABLE_PRIME_SIZE {
		log.Fatalf("intervals length (%d) != TABLE_PRIME_SIZE (%d). File: %s", len(publicTableC_prime), TABLE_PRIME_SIZE, intervalsPath)
	}

	// Define the intervals and coefficients vectors
	var (
		privateIndices_lookup     [][]int
		privateIndices_rangeProof [][]int
		Y_prime_ValSlice          = make([]uint64, NUM_INSTANCES)
		Y_ValSlice                = make([]uint64, NUM_INSTANCES)
		Delta_Val                 uint64
	)
	Delta_Val = DELTA_VALUE

	switch ACTIVATION {
	case "gelu":
		pairsPath := filepath.Join("../../../ZIP_lookup/examples", VALUES_DIR, "gelu_y_yprime.txt")
		Ys, Yps, thirds, err := loadPairsAndThirds(pairsPath, NUM_INSTANCES)
		if err != nil {
			log.Fatalf("failed to load Y/Y'/third triples: %v", err)
		}
		privateIndices_lookup, privateIndices_rangeProof, err = buildIndicesFromThirds(thirds)
		if err != nil {
			log.Fatalf("failed to build indices from thirds: %v", err)
		}
		for i := 0; i < NUM_INSTANCES; i++ {
			Y_ValSlice[i] = Ys[i]
			Y_prime_ValSlice[i] = Yps[i]
		}

	case "selu":
		pairsPath := filepath.Join("../../../ZIP_lookup/examples", VALUES_DIR, "selu_y_yprime.txt")
		Ys, Yps, thirds, err := loadPairsAndThirds(pairsPath, NUM_INSTANCES)
		if err != nil {
			log.Fatalf("failed to load Y/Y'/third triples: %v", err)
		}
		privateIndices_lookup, privateIndices_rangeProof, err = buildIndicesFromThirds(thirds)
		if err != nil {
			log.Fatalf("failed to build indices from thirds: %v", err)
		}
		for i := 0; i < NUM_INSTANCES; i++ {
			Y_ValSlice[i] = Ys[i]
			Y_prime_ValSlice[i] = Yps[i]
		}

	case "elu":
		pairsPath := filepath.Join("../../../ZIP_lookup/examples", VALUES_DIR, "elu_y_yprime.txt")
		Ys, Yps, thirds, err := loadPairsAndThirds(pairsPath, NUM_INSTANCES)
		if err != nil {
			log.Fatalf("failed to load Y/Y'/third triples: %v", err)
		}
		privateIndices_lookup, privateIndices_rangeProof, err = buildIndicesFromThirds(thirds)
		if err != nil {
			log.Fatalf("failed to build indices from thirds: %v", err)
		}
		for i := 0; i < NUM_INSTANCES; i++ {
			Y_ValSlice[i] = Ys[i]
			Y_prime_ValSlice[i] = Yps[i]
		}

	case "softmax":
		pairsPath := filepath.Join("../../../ZIP_lookup/examples", VALUES_DIR, "softmax_y_yprime.txt")
		Ys, Yps, thirds, err := loadPairsAndThirds(pairsPath, NUM_INSTANCES)
		if err != nil {
			log.Fatalf("failed to load Y/Y'/third triples: %v", err)
		}
		privateIndices_lookup, privateIndices_rangeProof, err = buildIndicesFromThirds(thirds)
		if err != nil {
			log.Fatalf("failed to build indices from thirds: %v", err)
		}
		for i := 0; i < NUM_INSTANCES; i++ {
			Y_ValSlice[i] = Ys[i]
			Y_prime_ValSlice[i] = Yps[i]
		}

	case "layernorm":
		pairsPath := filepath.Join("../../../ZIP_lookup/examples", VALUES_DIR, "layernorm_y_yprime.txt")
		Ys, Yps, thirds, err := loadPairsAndThirds(pairsPath, NUM_INSTANCES)
		if err != nil {
			log.Fatalf("failed to load Y/Y'/third triples: %v", err)
		}
		privateIndices_lookup, privateIndices_rangeProof, err = buildIndicesFromThirds(thirds)
		if err != nil {
			log.Fatalf("failed to build indices from thirds: %v", err)
		}
		for i := 0; i < NUM_INSTANCES; i++ {
			Y_ValSlice[i] = Ys[i]
			Y_prime_ValSlice[i] = Yps[i]
		}

	default:
		log.Fatalf("Unsupported ACTIVATION: %q", ACTIVATION)
	}

	// Convert the slices to fixed-size arrays.
	var Y_prime_ValArr [NUM_INSTANCES]uint64
	var Y_ValArr [NUM_INSTANCES]uint64
	copy(Y_prime_ValArr[:], Y_prime_ValSlice)
	copy(Y_ValArr[:], Y_ValSlice)

	var Y_prime_ValArrFixed [NUM_INSTANCES]frontend.Variable
	var Y_ValArrFixed [NUM_INSTANCES]frontend.Variable
	for i, v := range Y_prime_ValArr {
		Y_prime_ValArrFixed[i] = frontend.Variable(v)
	}
	for i, v := range Y_ValArr {
		Y_ValArrFixed[i] = frontend.Variable(v)
	}

	var cVecWitness [TABLE_SIZE]frontend.Variable
	for i, v := range publicTableC {
		cVecWitness[i] = frontend.Variable(v)
	}

	var c_primeVecWitness [TABLE_PRIME_SIZE]frontend.Variable
	for i, v := range publicTableC_prime {
		c_primeVecWitness[i] = frontend.Variable(v)
	}

	var aVecWitness [NUM_INSTANCES][PRIVATE_VECTOR_SIZE]frontend.Variable
	var sVecWitness [NUM_INSTANCES][PRIVATE_VECTOR_SIZE][TABLE_SIZE]frontend.Variable
	var a_primeVecWitness [NUM_INSTANCES][PRIVATE_VECTOR_PRIME_SIZE]frontend.Variable
	var s_primeVecWitness [NUM_INSTANCES][PRIVATE_VECTOR_PRIME_SIZE][TABLE_PRIME_SIZE]frontend.Variable

	for instance := 0; instance < NUM_INSTANCES; instance++ {

		// ---- coefficients (table C) ----

		privateTableA := make([]uint64, PRIVATE_VECTOR_SIZE)
		for j, idx := range privateIndices_lookup[instance] {
			privateTableA[j] = publicTableC[idx]
		}
		for j, v := range privateTableA {
			aVecWitness[instance][j] = frontend.Variable(v)
		}
		for j, idx := range privateIndices_lookup[instance] {
			for i := 0; i < TABLE_SIZE; i++ {
				if i == idx {
					sVecWitness[instance][j][i] = frontend.Variable(1)
				} else {
					sVecWitness[instance][j][i] = frontend.Variable(0)
				}
			}
		}

		// ---- intervals (table C') ----

		privateTableA_prime := make([]uint64, PRIVATE_VECTOR_PRIME_SIZE)
		for j, idx := range privateIndices_rangeProof[instance] {
			privateTableA_prime[j] = publicTableC_prime[idx]
		}
		for j, v := range privateTableA_prime {
			a_primeVecWitness[instance][j] = frontend.Variable(v)
		}
		for j, idx := range privateIndices_rangeProof[instance] {
			for i := 0; i < TABLE_PRIME_SIZE; i++ {
				if i == idx {
					s_primeVecWitness[instance][j][i] = frontend.Variable(1)
				} else {
					s_primeVecWitness[instance][j][i] = frontend.Variable(0)
				}
			}
		}
	}

	circuit := &FloatCircuit{ //F64
		E:    E_VALUE,
		M:    M_VALUE,
		Size: SIZE_VALUE,
	}

	// 4) Compile using the R1CS builder (for Groth16).
	r1csCircuit, err := frontend.Compile(ecc.BN254.ScalarField(), r1cs.NewBuilder, circuit)
	if err != nil {
		log.Fatalf("compile error: %v", err)
	}
	totalConstraints := r1csCircuit.GetNbConstraints()

	if EVAL_MODE == "fig1" {
		constraintsPerInstance := totalConstraints / NUM_INSTANCES
		fmt.Printf("Total number of R1CS constraints per instance (lower bound): %d\n", constraintsPerInstance)
	} else if EVAL_MODE == "table1/2" {
		fmt.Printf("ZIP/Table 1 - Total number of R1CS constraints: %d\n", totalConstraints)
	} else if EVAL_MODE == "table" {
		fmt.Printf("ZIP - Total number of R1CS constraints: %d\n", totalConstraints)
	}

	// 4) Compile
	ccs, err := frontend.Compile(ecc.BN254.ScalarField(), scs.NewBuilder, circuit)
	if err != nil {
		log.Fatalf("compile error: %v", err)
	}

	if EVAL_MODE == "fig1" {
		constraints := ccs.GetNbConstraints() / NUM_INSTANCES
		fmt.Println("Total number of PlonK constraints per instance (lower bound):", constraints)
	} else if EVAL_MODE == "table1/2" {
		fmt.Println("ZIP/Table 2 - Total number of PlonK constraints:", ccs.GetNbConstraints())
	} else if EVAL_MODE == "table" {
		fmt.Println("ZIP - Total number of PlonK constraints:", ccs.GetNbConstraints())
	}

	// 5) Create witness
	witnessFull := &FloatCircuit{
		//A:           aVal,
		//B:           bVal,
		//C:           cVal,
		E:           E_VALUE,
		M:           M_VALUE,
		Size:        SIZE_VALUE,
		A_vec:       aVecWitness,
		S_vec:       sVecWitness,
		C_vec:       cVecWitness,
		A_prime_vec: a_primeVecWitness,
		S_prime_vec: s_primeVecWitness,
		C_prime_vec: c_primeVecWitness,
		Y_prime:     Y_prime_ValArrFixed,
		Y:           Y_ValArrFixed,
		Delta:       Delta_Val,
	}
	witnessPub := &FloatCircuit{
		//C:           cVal,
		E:           E_VALUE,
		M:           M_VALUE,
		Size:        SIZE_VALUE,
		C_vec:       cVecWitness,
		C_prime_vec: c_primeVecWitness,
		Delta:       Delta_Val,
	}

	fw, _ := frontend.NewWitness(witnessFull, ecc.BN254.ScalarField())
	pw, _ := frontend.NewWitness(witnessPub, ecc.BN254.ScalarField(), frontend.PublicOnly())

	// 6) Cast to *cs.SparseR1CS
	if PROVING {
		// 7) Setup SRS
		srs, srsLagrange, err := unsafekzg.NewSRS(ccs)
		if err != nil {
			log.Fatal(err)
		}

		pk, vk, err := plonk.Setup(ccs, srs, srsLagrange)
		if err != nil {
			log.Fatalf("setup error: %v", err)
		}

		// 8) Prove
		tProve := time.Now()
		proof, err := plonk.Prove(ccs, pk, fw)
		if err != nil {
			log.Fatalf("prove error: %v", err)
		}
		proveSecs := time.Since(tProve).Seconds()
		fmt.Printf("Proving time: %.3f sec\n", proveSecs)

		var buf bytes.Buffer
		_, err = proof.WriteTo(&buf)
		if err != nil {
			log.Fatalf("failed to write proof: %v", err)
		}

		// 9) Verify
		tVerify := time.Now()
		if err := plonk.Verify(proof, vk, pw); err != nil {
			log.Fatalf("verify error: %v", err)
		}
		verifySecs := time.Since(tVerify).Seconds()
		fmt.Printf("Verification time: %.3f sec\n", verifySecs)

		fmt.Printf("Proof size: %d bytes\n", buf.Len())

		fmt.Println("Proof verified successfully using PLONK with float library!")

		timesPath := "../../../../../../proof_times.txt"
		if v := os.Getenv("ZIP_TIMES_FILE"); v != "" {
			timesPath = v
		}
		if err := os.MkdirAll(filepath.Dir(timesPath), 0o755); err != nil {
			log.Printf("warn: cannot create parent dir for %s: %v", timesPath, err)
		} else {
			f, err := os.OpenFile(timesPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
			if err != nil {
				log.Printf("warn: cannot open %s: %v", timesPath, err)
			} else {
				fmt.Fprintf(f, "%.6f, %.6f\n", proveSecs, verifySecs)
				_ = f.Close()
			}
		}
	}
}
