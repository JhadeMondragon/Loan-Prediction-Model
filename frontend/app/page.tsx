"use client"

import { useState } from "react"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipProvider, TooltipTrigger } from "@/components/ui/tooltip"
import { Info } from "lucide-react"

// Define input types
interface FormData {
  Age: string | number
  Income: string | number
  LoanAmount: string | number
  CreditScore: string | number
  MonthsEmployed: string | number
  NumCreditLines: string | number
  InterestRate: string | number
  LoanTerm: string | number
  DTIRatio: string | number
  Education: string
  EmploymentType: string
  MaritalStatus: string
  HasMortgage: string
  HasDependents: string
  LoanPurpose: string
  HasCoSigner: string
}

const initialData: FormData = {
  Age: "30",
  Income: "65000",
  LoanAmount: "15000",
  CreditScore: "720",
  MonthsEmployed: "24",
  NumCreditLines: "3",
  InterestRate: "4.5",
  LoanTerm: "36",
  DTIRatio: "0.25",
  Education: "Bachelor's",
  EmploymentType: "Full-time",
  MaritalStatus: "Single",
  HasMortgage: "No",
  HasDependents: "No",
  LoanPurpose: "Auto",
  HasCoSigner: "No"
}

export default function Home() {
  const [formData, setFormData] = useState<FormData>(initialData)
  const [prediction, setPrediction] = useState<any>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState("")
  const [fieldErrors, setFieldErrors] = useState<Partial<Record<keyof FormData, string>>>({})

  const validateForm = () => {
    const newErrors: Partial<Record<keyof FormData, string>> = {}
    let isValid = true

    const age = Number(formData.Age)
    if (age < 18) { newErrors.Age = "Must be at least 18"; isValid = false }
    if (age > 100) { newErrors.Age = "Must be realistic"; isValid = false }

    const income = Number(formData.Income)
    if (income <= 0) { newErrors.Income = "Must be greater than 0"; isValid = false }

    const loanAmount = Number(formData.LoanAmount)
    if (loanAmount <= 0) { newErrors.LoanAmount = "Must be greater than 0"; isValid = false }

    const creditScore = Number(formData.CreditScore)
    if (creditScore < 300 || creditScore > 850) { newErrors.CreditScore = "Must be between 300 and 850"; isValid = false }

    const monthsEmployed = Number(formData.MonthsEmployed)
    if (monthsEmployed < 0) { newErrors.MonthsEmployed = "Cannot be negative"; isValid = false }

    const numCreditLines = Number(formData.NumCreditLines)
    if (numCreditLines < 0) { newErrors.NumCreditLines = "Cannot be negative"; isValid = false }

    const interestRate = Number(formData.InterestRate)
    if (interestRate <= 0) { newErrors.InterestRate = "Must be positive"; isValid = false }
    if (interestRate > 100) { newErrors.InterestRate = "Too high"; isValid = false }

    const loanTerm = Number(formData.LoanTerm)
    if (loanTerm <= 0) { newErrors.LoanTerm = "Must be positive"; isValid = false }

    const dtiRatio = Number(formData.DTIRatio)
    if (dtiRatio < 0 || dtiRatio > 1) { newErrors.DTIRatio = "Must be between 0 and 1"; isValid = false }

    setFieldErrors(newErrors)
    return isValid
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    // Clear error when user types
    if (fieldErrors[name as keyof FormData]) {
      setFieldErrors(prev => ({ ...prev, [name]: undefined }))
    }
  }

  const handleSelectChange = (name: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!validateForm()) {
      return
    }

    setLoading(true)
    setError("")
    // Short delay to show animation
    await new Promise(r => setTimeout(r, 600))

    try {
      const payload = {
        ...formData,
        Age: Number(formData.Age),
        Income: Number(formData.Income),
        LoanAmount: Number(formData.LoanAmount),
        CreditScore: Number(formData.CreditScore),
        MonthsEmployed: Number(formData.MonthsEmployed),
        NumCreditLines: Number(formData.NumCreditLines),
        InterestRate: Number(formData.InterestRate),
        LoanTerm: Number(formData.LoanTerm),
        DTIRatio: Number(formData.DTIRatio),
      }

      const res = await fetch('http://localhost:8000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
      })

      if (!res.ok) {
        throw new Error(`Server error: ${res.statusText}`)
      }

      const data = await res.json()
      setPrediction(data)
    } catch (err: any) {
      setError(err.message || "Failed to fetch prediction")
    } finally {
      setLoading(false)
    }
  }

  // Helper to determine color based on risk
  const getRiskColor = (prob: number) => {
    if (prob < 0.2) return "text-emerald-400"
    if (prob < 0.5) return "text-yellow-400"
    return "text-rose-500"
  }

  // Helper for info tooltips
  const InfoIcon = () => <Info className="h-3 w-3 text-slate-600 hover:text-blue-400 cursor-help transition-colors" />

  return (
    <div className="dark min-h-screen w-full bg-slate-950 text-slate-100 p-4 md:p-8 flex flex-col items-center justify-center font-sans">

      {/* Cool Background Glow */}
      <div className="fixed top-0 left-0 w-full h-full overflow-hidden -z-10 pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-blue-900/20 rounded-full blur-[100px]" />
        <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-900/20 rounded-full blur-[100px]" />
      </div>

      <TooltipProvider>
        <div className="max-w-6xl w-full grid grid-cols-1 lg:grid-cols-3 gap-8">

          {/* Input Section */}
          <div className="lg:col-span-2">
            <Card className="border-slate-800 bg-slate-900/80 backdrop-blur-md shadow-xl">
              <CardHeader className="border-b border-slate-800 pb-6">
                <CardTitle className="text-2xl font-bold tracking-tight text-white flex items-center gap-2">
                  <span className="h-3 w-3 rounded-full bg-blue-500 animate-pulse"></span>
                  Loan Default Prediction Model
                </CardTitle>
                <CardDescription className="text-slate-400">
                  Advanced credit risk assessment system.
                </CardDescription>
              </CardHeader>

              <CardContent className="p-6">
                <form onSubmit={handleSubmit} className="space-y-8">

                  {/* Financial Metrics */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="h-px flex-1 bg-slate-800"></div>
                      <span className="text-xs font-mono text-blue-400 uppercase tracking-widest">Financial Metrics</span>
                      <div className="h-px flex-1 bg-slate-800"></div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
                      <div className="space-y-2 group">
                        <div className="flex items-center gap-1.5">
                          <Label htmlFor="Age" className="text-xs text-slate-500 group-hover:text-blue-400 transition-colors">Age</Label>
                        </div>
                        <Input id="Age" name="Age" type="number" className={`bg-slate-950 border-slate-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-mono ${fieldErrors.Age ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`} value={formData.Age} onChange={handleChange} />
                        {fieldErrors.Age && <span className="text-xs text-red-500">{fieldErrors.Age}</span>}
                      </div>
                      <div className="space-y-2 group">
                        <div className="flex items-center gap-1.5">
                          <Label htmlFor="Income" className="text-xs text-slate-500 group-hover:text-blue-400 transition-colors">Income ($)</Label>
                          <Tooltip>
                            <TooltipTrigger><InfoIcon /></TooltipTrigger>
                            <TooltipContent><p>Annual pre-tax earnings.</p></TooltipContent>
                          </Tooltip>
                        </div>
                        <Input id="Income" name="Income" type="number" className={`bg-slate-950 border-slate-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-mono ${fieldErrors.Income ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`} value={formData.Income} onChange={handleChange} />
                        {fieldErrors.Income && <span className="text-xs text-red-500">{fieldErrors.Income}</span>}
                      </div>
                      <div className="space-y-2 group">
                        <div className="flex items-center gap-1.5">
                          <Label htmlFor="LoanAmount" className="text-xs text-slate-500 group-hover:text-blue-400 transition-colors">Loan Amt ($)</Label>
                          <Tooltip>
                            <TooltipTrigger><InfoIcon /></TooltipTrigger>
                            <TooltipContent><p>Total principal amount requested.</p></TooltipContent>
                          </Tooltip>
                        </div>
                        <Input id="LoanAmount" name="LoanAmount" type="number" className={`bg-slate-950 border-slate-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-mono ${fieldErrors.LoanAmount ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`} value={formData.LoanAmount} onChange={handleChange} />
                        {fieldErrors.LoanAmount && <span className="text-xs text-red-500">{fieldErrors.LoanAmount}</span>}
                      </div>
                      <div className="space-y-2 group">
                        <div className="flex items-center gap-1.5">
                          <Label htmlFor="CreditScore" className="text-xs text-slate-500 group-hover:text-blue-400 transition-colors">FICO Score</Label>
                          <Tooltip>
                            <TooltipTrigger><InfoIcon /></TooltipTrigger>
                            <TooltipContent className="max-w-[200px]">
                              <p>A credit score ranging from 300 to 850. Higher scores indicate lower risk to lenders.</p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <Input id="CreditScore" name="CreditScore" type="number" className={`bg-slate-950 border-slate-800 focus:border-blue-500 focus:ring-1 focus:ring-blue-500 transition-all font-mono ${fieldErrors.CreditScore ? 'border-red-500 focus:border-red-500 focus:ring-red-500' : ''}`} value={formData.CreditScore} onChange={handleChange} />
                        {fieldErrors.CreditScore && <span className="text-xs text-red-500">{fieldErrors.CreditScore}</span>}
                      </div>

                      <div className="space-y-2 group">
                        <div className="flex items-center gap-1.5">
                          <Label htmlFor="InterestRate" className="text-xs text-slate-500 group-hover:text-blue-400 transition-colors">Interest %</Label>
                          <Tooltip>
                            <TooltipTrigger><InfoIcon /></TooltipTrigger>
                            <TooltipContent><p>Annual Percentage Rate (APR) charged.</p></TooltipContent>
                          </Tooltip>
                        </div>
                        <Input id="InterestRate" name="InterestRate" type="number" step="0.1" className={`bg-slate-950 border-slate-800 focus:border-blue-500 font-mono ${fieldErrors.InterestRate ? 'border-red-500 focus:border-red-500' : ''}`} value={formData.InterestRate} onChange={handleChange} />
                        {fieldErrors.InterestRate && <span className="text-xs text-red-500">{fieldErrors.InterestRate}</span>}
                      </div>
                      <div className="space-y-2 group">
                        <div className="flex items-center gap-1.5">
                          <Label htmlFor="DTIRatio" className="text-xs text-slate-500 group-hover:text-blue-400 transition-colors">DTI (0-1)</Label>
                          <Tooltip>
                            <TooltipTrigger><InfoIcon /></TooltipTrigger>
                            <TooltipContent className="max-w-[200px]">
                              <p>Debt-to-Income Ratio. The percentage of your monthly income that goes toward paying debts. Lower (e.g., 0.3) is better.</p>
                            </TooltipContent>
                          </Tooltip>
                        </div>
                        <Input id="DTIRatio" name="DTIRatio" type="number" step="0.01" className={`bg-slate-950 border-slate-800 focus:border-blue-500 font-mono ${fieldErrors.DTIRatio ? 'border-red-500 focus:border-red-500' : ''}`} value={formData.DTIRatio} onChange={handleChange} />
                        {fieldErrors.DTIRatio && <span className="text-xs text-red-500">{fieldErrors.DTIRatio}</span>}
                      </div>
                      <div className="space-y-2 group">
                        <div className="flex items-center gap-1.5">
                          <Label htmlFor="LoanTerm" className="text-xs text-slate-500 group-hover:text-blue-400 transition-colors">Term (Mos)</Label>
                          <Tooltip>
                            <TooltipTrigger><InfoIcon /></TooltipTrigger>
                            <TooltipContent><p>Duration of the loan in months.</p></TooltipContent>
                          </Tooltip>
                        </div>
                        <Input id="LoanTerm" name="LoanTerm" type="number" className={`bg-slate-950 border-slate-800 focus:border-blue-500 font-mono ${fieldErrors.LoanTerm ? 'border-red-500 focus:border-red-500' : ''}`} value={formData.LoanTerm} onChange={handleChange} />
                        {fieldErrors.LoanTerm && <span className="text-xs text-red-500">{fieldErrors.LoanTerm}</span>}
                      </div>
                      <div className="space-y-2 group">
                        <div className="flex items-center gap-1.5">
                          <Label htmlFor="NumCreditLines" className="text-xs text-slate-500 group-hover:text-blue-400 transition-colors">Lines</Label>
                          <Tooltip>
                            <TooltipTrigger><InfoIcon /></TooltipTrigger>
                            <TooltipContent><p>Number of active credit lines (cards, loans) currently open.</p></TooltipContent>
                          </Tooltip>
                        </div>
                        <Input id="NumCreditLines" name="NumCreditLines" type="number" className={`bg-slate-950 border-slate-800 focus:border-blue-500 font-mono ${fieldErrors.NumCreditLines ? 'border-red-500 focus:border-red-500' : ''}`} value={formData.NumCreditLines} onChange={handleChange} />
                        {fieldErrors.NumCreditLines && <span className="text-xs text-red-500">{fieldErrors.NumCreditLines}</span>}
                      </div>
                    </div>
                  </div>

                  {/* Personal Details */}
                  <div className="space-y-4">
                    <div className="flex items-center gap-2 mb-4">
                      <div className="h-px flex-1 bg-slate-800"></div>
                      <span className="text-xs font-mono text-blue-400 uppercase tracking-widest">Personal Profile</span>
                      <div className="h-px flex-1 bg-slate-800"></div>
                    </div>

                    <div className="grid grid-cols-2 md:grid-cols-4 gap-5">

                      <div className="space-y-2">
                        <Label className="text-xs text-slate-500">Education</Label>
                        <Select onValueChange={(val) => handleSelectChange('Education', val)} value={formData.Education}>
                          <SelectTrigger className="bg-slate-950 border-slate-800 text-slate-300"><SelectValue /></SelectTrigger>
                          <SelectContent className="bg-slate-900 border-slate-800 text-slate-300">
                            <SelectItem value="High School">High School</SelectItem>
                            <SelectItem value="Bachelor's">Bachelor's</SelectItem>
                            <SelectItem value="Master's">Master's</SelectItem>
                            <SelectItem value="PhD">PhD</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-slate-500">Employment</Label>
                        <Select onValueChange={(val) => handleSelectChange('EmploymentType', val)} value={formData.EmploymentType}>
                          <SelectTrigger className="bg-slate-950 border-slate-800 text-slate-300"><SelectValue /></SelectTrigger>
                          <SelectContent className="bg-slate-900 border-slate-800 text-slate-300">
                            <SelectItem value="Full-time">Full-time</SelectItem>
                            <SelectItem value="Part-time">Part-time</SelectItem>
                            <SelectItem value="Unemployed">Unemployed</SelectItem>
                            <SelectItem value="Self-employed">Self-employed</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-slate-500">Marital</Label>
                        <Select onValueChange={(val) => handleSelectChange('MaritalStatus', val)} value={formData.MaritalStatus}>
                          <SelectTrigger className="bg-slate-950 border-slate-800 text-slate-300"><SelectValue /></SelectTrigger>
                          <SelectContent className="bg-slate-900 border-slate-800 text-slate-300">
                            <SelectItem value="Single">Single</SelectItem>
                            <SelectItem value="Married">Married</SelectItem>
                            <SelectItem value="Divorced">Divorced</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-slate-500">Purpose</Label>
                        <Select onValueChange={(val) => handleSelectChange('LoanPurpose', val)} value={formData.LoanPurpose}>
                          <SelectTrigger className="bg-slate-950 border-slate-800 text-slate-300"><SelectValue /></SelectTrigger>
                          <SelectContent className="bg-slate-900 border-slate-800 text-slate-300">
                            <SelectItem value="Home">Home</SelectItem>
                            <SelectItem value="Auto">Auto</SelectItem>
                            <SelectItem value="Education">Education</SelectItem>
                            <SelectItem value="Business">Business</SelectItem>
                            <SelectItem value="Other">Other</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-slate-500">Mortgage</Label>
                        <Select onValueChange={(val) => handleSelectChange('HasMortgage', val)} value={formData.HasMortgage}>
                          <SelectTrigger className="bg-slate-950 border-slate-800 text-slate-300"><SelectValue /></SelectTrigger>
                          <SelectContent className="bg-slate-900 border-slate-800 text-slate-300">
                            <SelectItem value="Yes">Yes</SelectItem>
                            <SelectItem value="No">No</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>

                      <div className="space-y-2">
                        <Label className="text-xs text-slate-500">Dependents</Label>
                        <Select onValueChange={(val) => handleSelectChange('HasDependents', val)} value={formData.HasDependents}>
                          <SelectTrigger className="bg-slate-950 border-slate-800 text-slate-300"><SelectValue /></SelectTrigger>
                          <SelectContent className="bg-slate-900 border-slate-800 text-slate-300">
                            <SelectItem value="Yes">Yes</SelectItem>
                            <SelectItem value="No">No</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                      <div className="space-y-2">
                        <Label className="text-xs text-slate-500">Co-Signer</Label>
                        <Select onValueChange={(val) => handleSelectChange('HasCoSigner', val)} value={formData.HasCoSigner}>
                          <SelectTrigger className="bg-slate-950 border-slate-800 text-slate-300"><SelectValue /></SelectTrigger>
                          <SelectContent className="bg-slate-900 border-slate-800 text-slate-300">
                            <SelectItem value="Yes">Yes</SelectItem>
                            <SelectItem value="No">No</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                  </div>

                  <Button
                    type="submit"
                    className="w-full bg-blue-600 hover:bg-blue-500 text-white font-semibold tracking-wide py-6 shadow-[0_0_20px_rgba(37,99,235,0.3)] transition-all hover:scale-[1.01] active:scale-[0.99]"
                    disabled={loading}
                  >
                    {loading ? (
                      <span className="flex items-center gap-2">
                        <span className="h-4 w-4 border-2 border-white/30 border-t-white rounded-full animate-spin"></span>
                        PROCESSING NEURAL PATHWAYS...
                      </span>
                    ) : "INITIATE RISK ANALYSIS"}
                  </Button>

                </form>
              </CardContent>
            </Card>
          </div>

          {/* Results Dashboard */}
          <div className="lg:col-span-1 space-y-6">

            {/* Main Result Card */}
            <Card className="border-slate-800 bg-black/40 backdrop-blur-xl h-full flex flex-col items-center justify-center p-6 relative overflow-hidden">
              {!prediction && !loading && (
                <div className="text-center space-y-4 opacity-50">
                  <div className="w-24 h-24 rounded-full border-2 border-slate-800 flex items-center justify-center mx-auto">
                    <span className="text-4xl">⚠️</span>
                  </div>
                  <p className="text-slate-500 font-mono text-sm">Awaiting Data Input</p>
                </div>
              )}

              {loading && (
                <div className="text-center space-y-6">
                  <div className="relative w-32 h-32 mx-auto">
                    <div className="absolute inset-0 border-4 border-slate-800 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-blue-500 border-t-transparent rounded-full animate-spin"></div>
                    <div className="absolute inset-4 border-4 border-indigo-500/30 border-b-transparent rounded-full animate-spin direction-reverse"></div>
                  </div>
                  <p className="text-blue-400 font-mono text-sm animate-pulse">Running Inference...</p>
                </div>
              )}

              {prediction && !loading && (
                <div className="w-full h-full flex flex-col items-center animate-in fade-in zoom-in duration-500">
                  {/* Status Badge */}
                  <div className={`px-4 py-1 rounded-full border ${prediction.prediction === 1 ? 'border-rose-900/50 bg-rose-950/30 text-rose-400' : 'border-emerald-900/50 bg-emerald-950/30 text-emerald-400'} text-xs font-mono mb-8 uppercase tracking-widest`}>
                    {prediction.status}
                  </div>

                  {/* Gauge Visualization */}
                  <div className="relative w-48 h-48 mb-6 flex items-center justify-center">
                    {/* Background Circle */}
                    <svg className="absolute w-full h-full rotate-[-90deg]" viewBox="0 0 100 100">
                      <circle cx="50" cy="50" r="45" fill="none" stroke="#1e293b" strokeWidth="8" />
                      <circle
                        cx="50" cy="50" r="45" fill="none"
                        stroke="currentColor"
                        strokeWidth="8"
                        strokeDasharray="283"
                        strokeDashoffset={283 - (283 * prediction.probability)}
                        className={`${getRiskColor(prediction.probability)} transition-all duration-1000 ease-out`}
                        strokeLinecap="round"
                      />
                    </svg>

                    <div className="text-center z-10">
                      <span className={`text-4xl font-bold ${getRiskColor(prediction.probability)}`}>
                        {(prediction.probability * 100).toFixed(0)}
                        <span className="text-base align-top opacity-50">%</span>
                      </span>
                      <p className="text-xs text-slate-500 font-mono mt-1">PROBABILITY</p>
                    </div>
                  </div>

                  {/* Decision */}
                  <div className="text-center space-y-2 w-full mt-4">
                    <div className="flex justify-between text-sm text-slate-400 border-b border-slate-800 pb-2">
                      <span>Verdict</span>
                      <span className={prediction.prediction === 1 ? "text-rose-400 font-bold" : "text-emerald-400 font-bold"}>
                        {prediction.prediction === 1 ? "DENY" : "APPROVE"}
                      </span>
                    </div>
                    <div className="flex justify-between text-sm text-slate-400 border-b border-slate-800 pb-2">
                      <span>Confidence</span>
                      <span className="text-white">High</span>
                    </div>
                    <div className="flex justify-between text-sm text-slate-400 border-b border-slate-800 pb-2">
                      <span>Model</span>
                      <span className="text-blue-400">XGBoost v2</span>
                    </div>
                  </div>
                </div>
              )}
            </Card>

          </div>
        </div>
      </TooltipProvider>
    </div>
  )
}
