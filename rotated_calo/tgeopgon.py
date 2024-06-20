import ROOT

def main():
    pgon = ROOT.TGeoPgon("pg", 0., 360., 12, 2)
    pgon.DefineSection(0, 0, 10, 20)
    pgon.DefineSection(1, 2, 10, 20)

    canv = ROOT.TCanvas("canv", "canv", 800, 800)
    pgon.Draw()
    canv.SaveAs("tgeopgon.pdf")


if __name__ == '__main__':
    main()
